"""LangChain 1.0 Human-in-the-loop 미들웨어 데모."""

from __future__ import annotations

import os
from typing import Any, Sequence, cast

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

# LangChain 모델이 도구 사용 시 따라야 할 안전 규칙을 정의한다.
DEFAULT_SYSTEM_PROMPT = (
    "당신은 사용자의 이름을 안전하게 저장하는 비서입니다. "
    "`save_user_name` 도구를 사용하기 전에는 이름을 기억했다고 주장하지 마세요. "
    "사용자가 이름을 말하면 반드시 해당 도구를 호출해 승인을 기다리세요."
)

# HumanInTheLoopMiddleware가 감시할 툴과 허용할 결정 타입을 선언한다.
DEFAULT_INTERRUPT_ON: dict[str, bool | InterruptOnConfig] = {
    "save_user_name": {
        "allowed_decisions": ["approve", "reject"],
        "description": "사용자 이름 저장 요청. 실행 전 승인 또는 거절이 필요합니다.",
    }
}

# 타입 안정성을 위해 create_agent가 반환하는 LangGraph 에이전트 유형을 별칭으로 둔다.
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

    # HumanInTheLoopMiddleware는 LangGraph 노드 실행 전 툴 호출을 가로채 사용자 결정을 요구한다.
    middleware = HumanInTheLoopMiddleware(interrupt_on=resolved_interrupt_on)

    agent: AgentGraph = create_agent(
        model=model,
        tools=[save_user_name],
        system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
        middleware=[
            middleware
        ],  # LangGraph 체크포인터는 인터럽트 시점의 그래프 상태를 저장하고, 사용자의 수동 결정을 기다린 뒤 재개한다.
        checkpointer=checkpointer,
    )
    return agent


def prompt_user_decision(request: HITLRequest, summary: str) -> Decision:
    """사용자의 승인/거절 결정을 입력받는다.

    Args:
        request: 승인 대기 중인 요청 정보.

    Returns:
        Human-in-the-loop 재개에 필요한 결정 사전.
    """
    while True:
        prompt_text = (
            f"{summary}\n승인하시겠습니까? [a=approve / r=reject] (기본값: a) "
        )
        user_choice = input(prompt_text).strip().lower()

        # 사용자가 승인하면 approve 결정을 반환한다.
        if user_choice in ("", "a", "approve"):
            return {"type": "approve"}

        # 사용자가 거절하면 reject 결정을 반환한다.
        if user_choice in ("r", "reject"):
            rejection_reason = input(
                "거절 사유를 입력하세요 (엔터 시 기본 메시지): "
            ).strip()
            if rejection_reason:
                return cast(Decision, {"type": "reject", "message": rejection_reason})
            return {"type": "reject"}

        print("입력이 올바르지 않습니다. a 또는 r을 입력하세요.")


def _print_messages(messages: Sequence[BaseMessage], seen_ids: set[str]) -> None:
    """이미 출력된 메시지를 건너뛰며 AI/툴 메시지를 순서대로 출력한다."""
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


def _summarize_hitl_request(request: HITLRequest) -> str:
    """HITL 요청에 포함된 모든 액션을 문자열로 정리한다."""
    actions = request["action_requests"]
    if not actions:
        return "\n[승인 대기] 처리할 툴 요청이 없습니다."

    lines = ["\n[승인 대기] LangGraph가 다음 툴 실행을 기다리고 있습니다."]
    for idx, action in enumerate(actions, start=1):
        prefix = f"  ({idx})"
        lines.append(f"{prefix} 툴 이름: {action['name']}")
        lines.append(f"      인자: {action['args']}")
        description = action.get("description")
        if description:
            lines.append(f"      설명: {description}")
    lines.append("  옵션: a=승인, r=거절")
    return "\n".join(lines)


def run_hitl_flow(agent: AgentGraph, payload: Payload, config: RunnableConfig) -> None:
    """단일 루프에서 메시지 스트림과 인터럽트를 처리한다."""
    next_payload: Payload | Command = payload
    printed_message_ids: set[str] = set()

    while True:
        interrupted = False

        # agent.stream은 LangGraph 그래프 실행 결과를 이벤트로 스트리밍하므로, 메시지 출력과 인터럽트 재개 결정을 동일 루프 안에서 다룬다.
        for event in agent.stream(next_payload, config, stream_mode="values"):
            messages = event.get("messages") or []
            _print_messages(messages, printed_message_ids)

            interrupt_event = event.get(
                "__interrupt__"
            )  # LangGraph 인터럽트 이벤트를 확인한다.

            if interrupt_event:
                hitl_request = interrupt_event[0].value
                summary = _summarize_hitl_request(hitl_request)
                decision = prompt_user_decision(
                    hitl_request, summary
                )  # 사용자의 수동 결정을 입력 받는다.
                next_payload = Command(resume={"decisions": [decision]})
                interrupted = True
                break

        if not interrupted:
            break


def main() -> None:
    """데모 스크립트를 실행한다."""
    load_dotenv()
    model_name = os.environ.get("HITL_DEMO_MODEL", "gemini-2.5-flash")
    model = ChatGoogleGenerativeAI(model=model_name)

    user_name = input("저장할 이름을 입력하세요: ").strip() or "LangChain User"

    # LangChain invoke 입력을 사용자 이름과 도구 사용 지시를 별도 turn으로 구성한다.
    initial_payload = {
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
    # RunnableConfig의 thread_id는 LangGraph 체크포인터가 상태를 식별·복원할 때 사용된다.
    config: RunnableConfig = {"configurable": {"thread_id": "hitl-demo"}}

    with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
        agent = build_hitl_agent(model=model, checkpointer=checkpointer)
        run_hitl_flow(agent, initial_payload, config)


if __name__ == "__main__":
    main()
