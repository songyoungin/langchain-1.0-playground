from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver

# .env 파일을 불러 LangGraph 실행에 필요한 API 키를 준비한다.
load_dotenv()


def save_user_name(name: str) -> str:
    """사용자의 이름을 저장했다고 가정하고 확인 메시지를 반환한다.

    Args:
        name: 저장할 사용자 이름.

    Returns:
        이름을 기억했음을 알리는 안내 문장.
    """
    return f"I've remembered your name, {name}."


model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# LangGraph MemorySaver는 그래프 상태를 프로세스 메모리에 유지해 세션 간 맥락을 살린다.
memory = MemorySaver()

agent: Any = create_agent(
    model=model,
    tools=[save_user_name],
    # LangGraph 체크포인터에 연결하면 `thread_id` 별 메시지 그래프가 자동 저장된다.
    checkpointer=memory,
    system_prompt="You are a helpful assistant that remembers conversation history.",
)

print("=" * 80)
print("LangChain 1.0 - 기본 메모리 예제 (MemorySaver)")
print("=" * 80)
print()

print("[세션 1] thread_id: user-1")
print("-" * 80)

# LangGraph 실행 설정에서 thread_id는 동일 그래프 상태를 재사용할 키로 쓰인다.
config_1: dict[str, Any] = {"configurable": {"thread_id": "user-1"}}

print("\n[턴 1] 사용자 입력: 'My name is John'")
inputs_1: dict[str, Any] = {
    "messages": [{"role": "user", "content": "My name is John"}]
}
# LangGraph invoke는 메시지 배열을 받아 상태 그래프에 추가하고 응답을 생성한다.
result_1: dict[str, Any] = agent.invoke(inputs_1, config_1)
print(f"에이전트: {result_1['messages'][-1].content}")

print("\n[턴 2] 사용자 입력: 'What was my name?'")
inputs_2: dict[str, Any] = {
    "messages": [{"role": "user", "content": "What was my name?"}]
}
result_2: dict[str, Any] = agent.invoke(inputs_2, config_1)
print(f"에이전트: {result_2['messages'][-1].content}")

print("\n\n[세션 2] thread_id: user-2 (새 세션)")
print("-" * 80)

# 새로운 thread_id는 LangGraph가 별도 그래프 스냅샷을 생성하도록 해 세션을 격리한다.
config_2: dict[str, Any] = {"configurable": {"thread_id": "user-2"}}

print("\n[턴 1] 사용자 입력: 'What was my name?'")
inputs_3: dict[str, Any] = {
    "messages": [{"role": "user", "content": "What was my name?"}]
}
result_3: dict[str, Any] = agent.invoke(inputs_3, config_2)
print(f"에이전트: {result_3['messages'][-1].content}")
print("\n💡 세션이 분리되어 있어 이전 대화를 기억하지 않습니다.")

print("\n\n[세션 1 복귀] thread_id: user-1")
print("-" * 80)

# 동일 thread_id를 지정하면 이전에 저장된 LangGraph 상태가 로드되어 맥락이 이어진다.
print("\n[턴 3] 사용자 입력: 'Please tell me my name again'")
inputs_4: dict[str, Any] = {
    "messages": [{"role": "user", "content": "Please tell me my name again"}]
}
result_4: dict[str, Any] = agent.invoke(inputs_4, config_1)
print(f"에이전트: {result_4['messages'][-1].content}")
print("\n✅ 세션 1의 대화 이력이 그대로 유지되었습니다.")

print("\n" + "=" * 80)
print("메모리 기능 테스트를 완료했습니다!")
print("=" * 80)
print(
    "\n💡 참고: MemorySaver는 메모리 기반 저장소라 프로그램 종료 시 데이터가 사라집니다."
)
