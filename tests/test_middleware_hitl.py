"""Human-in-the-loop 미들웨어 예제 테스트."""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolCall, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from examples.middleware_hitl import build_hitl_agent


class ToolCallingReplayModel(BaseChatModel):
    """사전 정의된 AIMessage를 순차적으로 반환하는 테스트용 모델."""

    def __init__(self, responses: Sequence[AIMessage]) -> None:
        super().__init__()
        self._responses = list(responses)
        self._cursor = 0

    def bind_tools(self, tools, **kwargs):  # type: ignore[override]
        """툴 바인딩을 무시하고 자기 자신을 반환한다."""
        return self

    def _generate(  # type: ignore[override]
        self,
        messages,
        stop=None,
        run_manager=None,
        **kwargs,
    ) -> ChatResult:
        """사전 정의된 메시지를 순환하여 반환한다."""
        if not self._responses:
            msg = "responses 리스트가 비어 있습니다."
            raise ValueError(msg)
        response = self._responses[self._cursor]
        if self._cursor < len(self._responses) - 1:
            self._cursor += 1
        generation = ChatGeneration(message=response)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        return "tool-calling-replay"


def _build_agent_with_responses(responses: Sequence[AIMessage]):
    model = ToolCallingReplayModel(responses)
    agent = build_hitl_agent(
        model=model,
        checkpointer=MemorySaver(),
    )
    config = {"configurable": {"thread_id": "test-thread"}}
    return agent, config


def test_hitl_interrupt_contains_tool_info():
    """인터럽트 payload가 툴 이름과 인자를 포함하는지 확인한다."""
    tool_call = ToolCall(
        type="tool_call",
        name="save_user_name",
        args={"name": "Alice"},
        id="call_1",
    )
    agent, config = _build_agent_with_responses(
        [AIMessage(content="", tool_calls=[tool_call])]
    )

    interrupt_payload = None
    for event in agent.stream(
        {"messages": [HumanMessage(content="내 이름을 기억해")]},
        config,
        stream_mode="values",
    ):
        if "__interrupt__" in event:
            interrupt_payload = event["__interrupt__"][0].value
            break

    assert interrupt_payload is not None, "Human-in-the-loop 인터럽트가 발생해야 한다."
    assert interrupt_payload["action_requests"][0]["name"] == "save_user_name"
    assert interrupt_payload["action_requests"][0]["args"] == {"name": "Alice"}


def test_hitl_resume_leads_to_tool_execution():
    """승인 후 툴이 실행되고 응답 메시지가 생성되는지 확인한다."""
    tool_call = ToolCall(
        type="tool_call",
        name="save_user_name",
        args={"name": "Alice"},
        id="call_1",
    )
    responses = [
        AIMessage(content="", tool_calls=[tool_call]),
        AIMessage(content="승인이 완료되었습니다."),
    ]
    agent, config = _build_agent_with_responses(responses)

    stream = agent.stream(
        {"messages": [HumanMessage(content="내 이름을 기억해")]},
        config,
        stream_mode="values",
    )
    for event in stream:
        if "__interrupt__" in event:
            break

    resume_command = Command(resume={"decisions": [{"type": "approve"}]})
    final_messages: list | None = None
    for resume_event in agent.stream(resume_command, config, stream_mode="values"):
        if resume_event.get("messages"):
            final_messages = resume_event["messages"]

    assert final_messages, "재개 후 메시지가 생성되어야 한다."
    assert isinstance(final_messages[-1], AIMessage)
    assert final_messages[-1].content == "승인이 완료되었습니다."
    tool_messages = [msg for msg in final_messages if isinstance(msg, ToolMessage)]
    assert tool_messages, "툴 실행 결과 메시지가 필요하다."
    assert "Alice" in tool_messages[-1].content
