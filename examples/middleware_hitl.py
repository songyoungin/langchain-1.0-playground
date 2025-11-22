"""LangChain 1.0 Human-in-the-loop 미들웨어 데모."""

from __future__ import annotations

from collections.abc import Sequence
import os
from typing import Any, cast

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.agents.middleware.human_in_the_loop import (
    Decision,
    HITLRequest,
    InterruptOnConfig,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

DEFAULT_SYSTEM_PROMPT = (
    "당신은 사용자의 이름을 안전하게 저장하는 비서입니다. "
    "`save_user_name` 도구를 사용하기 전에는 이름을 기억했다고 주장하지 마세요. "
    "사용자가 이름을 말하면 반드시 해당 도구를 호출해 승인을 기다리세요."
)

DEFAULT_INTERRUPT_ON: dict[str, bool | InterruptOnConfig] = {
    "save_user_name": {
        "allowed_decisions": ["approve", "reject"],
        "description": "사용자 이름 저장 요청. 실행 전 승인 또는 거절이 필요합니다.",
    }
}

AgentGraph = CompiledStateGraph[Any, Any, Any, Any]
Payload = dict[str, Any]


def save_user_name(name: str) -> str:
    """사용자의 이름을 저장했다고 가정하고 확인 메시지를 반환한다.

    Args:
        name: 저장할 사용자 이름.

    Returns:
        사용자에게 저장 완료 사실을 알려 주는 한국어 문장.
    """
    return f"{name}님의 이름을 안전하게 저장했습니다."


def build_hitl_agent(
    *,
    model: BaseChatModel,
    checkpointer: BaseCheckpointSaver,
    interrupt_on: dict[str, bool | InterruptOnConfig] | None = None,
    system_prompt: str | None = None,
) -> AgentGraph:
    """Human-in-the-loop 미들웨어를 적용한 에이전트를 생성한다.

    Args:
        model: 에이전트가 사용할 언어 모델.
        checkpointer: LangGraph 인터럽트 처리를 위한 체크포인터.
        interrupt_on: 승인/거절이 필요한 툴 설정. 생략 시 기본값 사용.
        system_prompt: 모델 지침. 생략 시 기본 시스템 프롬프트 사용.

    Returns:
        Human-in-the-loop가 적용된 LangGraph 에이전트.
    """
    resolved_interrupt_on = interrupt_on or DEFAULT_INTERRUPT_ON
    # LangGraph HumanInTheLoopMiddleware가 툴 실행 노드를 인터럽트해 수동 결정을 기다리도록 한다.
    middleware = HumanInTheLoopMiddleware(interrupt_on=resolved_interrupt_on)
    agent: AgentGraph = create_agent(
        model=model,
        tools=[save_user_name],
        system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
        # 체크포인터를 연결하면 LangGraph가 인터럽트 지점과 메시지 그래프를 저장해 재개할 수 있다.
        middleware=[middleware],
        checkpointer=checkpointer,
    )
    return agent


def format_hitl_request(request: HITLRequest) -> str:
    """승인 대기 중인 툴 호출 정보를 보기 좋게 문자열로 만든다.

    Args:
        request: Human-in-the-loop 미들웨어가 전달한 요청 payload.

    Returns:
        CLI 출력에 사용할 설명 문자열.
    """
    lines = ["\n[알림] 다음 툴 실행에 대한 승인이 필요합니다:"]
    for idx, action in enumerate(request["action_requests"], start=1):
        lines.append(f"  ({idx}) 툴 이름: {action['name']}")
        lines.append(f"      인자: {action['args']}")
        description = action.get("description")
        if description:
            lines.append(f"      설명: {description}")
    lines.append("a=승인, r=거절 중 하나를 입력하세요.")
    return "\n".join(lines)


def prompt_user_decision(request: HITLRequest) -> Decision:
    """사용자의 승인/거절 결정을 입력받는다.

    Args:
        request: 승인 대기 중인 요청 정보.

    Returns:
        Human-in-the-loop 재개에 필요한 결정 사전.
    """
    while True:
        user_choice = (
            input("승인하시겠습니까? [a=approve / r=reject] (기본값: a) ")
            .strip()
            .lower()
        )
        if user_choice in ("", "a", "approve"):
            return {"type": "approve"}
        if user_choice in ("r", "reject"):
            rejection_reason = input(
                "거절 사유를 입력하세요 (엔터 시 기본 메시지): "
            ).strip()
            if rejection_reason:
                return cast(Decision, {"type": "reject", "message": rejection_reason})
            return {"type": "reject"}
        print("입력이 올바르지 않습니다. a 또는 r을 입력하세요.")


def _print_new_messages(messages: Sequence[BaseMessage], seen_ids: set[str]) -> None:
    """중복 없이 새로운 AI/툴 메시지를 출력한다."""
    for message in messages:
        message_id = getattr(message, "id", None)
        if message_id and message_id in seen_ids:
            continue

        if isinstance(message, ToolMessage):
            prefix = "[툴]"
        elif isinstance(message, AIMessage):
            prefix = "[에이전트]"
        else:
            continue

        content = message.content or ""
        print(f"{prefix} {content}")
        if message_id:
            seen_ids.add(message_id)


def stream_with_human_review(
    agent: AgentGraph,
    initial_payload: Payload,
    config: RunnableConfig,
) -> None:
    """Human-in-the-loop 인터럽트를 처리하며 결과를 출력한다.

    Args:
        agent: Human-in-the-loop 에이전트.
        initial_payload: 최초 사용자 입력 payload.
        config: LangGraph 실행 설정(스레드 ID 등 포함).
    """
    # LangGraph Command 객체를 재사용해 인터럽트 재개 결정과 사용자 메시지를 같은 루프로 처리한다.
    payload: Payload | Command = initial_payload
    printed_message_ids: set[str] = set()

    while True:
        interrupted = False
        # agent.stream은 LangGraph 상태 그래프를 이벤트 스트림으로 노출해 메시지와 인터럽트를 실시간 전달한다.
        for event in agent.stream(payload, config, stream_mode="values"):
            messages = event.get("messages")
            if messages:
                _print_new_messages(messages, printed_message_ids)

            interrupt_event = event.get("__interrupt__")
            if interrupt_event:
                hitl_request = interrupt_event[0].value
                print(format_hitl_request(hitl_request))
                decision = prompt_user_decision(hitl_request)
                # 사용자의 결정은 LangGraph Command(resume) 구조로 감싸 재개 지점의 노드에 전달된다.
                payload = Command(resume={"decisions": [decision]})
                interrupted = True
                break
        if not interrupted:
            break


def _create_initial_payload(user_name: str) -> Payload:
    """CLI 입력으로부터 LangChain 메시지 payload를 생성한다."""
    return {
        "messages": [
            {"role": "user", "content": f"내 이름은 {user_name}야."},
            {
                "role": "user",
                "content": (
                    "앞으로 나를 기억하기 위해 반드시 save_user_name 도구를 호출해 "
                    "이름을 저장해 줘."
                ),
            },
        ]
    }


def main() -> None:
    """데모 스크립트를 실행한다."""
    load_dotenv()
    model_name = os.environ.get("HITL_DEMO_MODEL", "gemini-2.5-flash")
    model = ChatGoogleGenerativeAI(model=model_name)

    user_name = input("저장할 이름을 입력하세요: ").strip() or "LangChain User"
    initial_payload = _create_initial_payload(user_name)
    config: RunnableConfig = {"configurable": {"thread_id": "hitl-demo"}}

    print(
        "\n[LangChain 1.0 HITL 데모]\n"
        "1. 모델이 도구를 호출하면 승인 여부를 직접 선택할 수 있습니다.\n"
        "2. 승인 시 저장 도구가 실행되고, 거절 시 거절 메시지가 모델에 전달됩니다.\n"
    )

    # SqliteSaver는 LangGraph 그래프 상태를 영속화해 인터럽트 중 프로세스가 종료돼도 재개할 수 있게 한다.
    with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
        agent = build_hitl_agent(model=model, checkpointer=checkpointer)
        stream_with_human_review(agent, initial_payload, config)


if __name__ == "__main__":
    main()
