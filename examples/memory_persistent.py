import os
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver

# .env 환경을 로드해 LangGraph 에이전트가 사용할 API 키를 준비한다.
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

# LangGraph SqliteSaver가 체크포인트를 기록할 데이터베이스 경로.
db_path = "checkpoints.db"

print("=" * 80)
print("LangChain 1.0 - 영속 메모리 예제 (SqliteSaver)")
print("=" * 80)
print(f"\n💾 데이터베이스 파일: {os.path.abspath(db_path)}")
print()

with SqliteSaver.from_conn_string(db_path) as checkpointer:
    agent: Any = create_agent(
        model=model,
        tools=[save_user_name],
        # SqliteSaver는 LangGraph 실행 스냅샷을 파일로 저장해 프로세스가 재시작해도 복원된다.
        checkpointer=checkpointer,
        system_prompt="You are a helpful assistant that permanently remembers conversation history.",
    )

    print("[세션 1] thread_id: persistent-user-1")
    print("-" * 80)

    # 동일 thread_id는 LangGraph가 같은 체크포인트 레코드를 갱신하도록 지시한다.
    config: dict[str, Any] = {"configurable": {"thread_id": "persistent-user-1"}}

    print("\n[턴 1] 사용자 입력: 'My name is Jane'")
    inputs_1: dict[str, Any] = {
        "messages": [{"role": "user", "content": "My name is Jane"}]
    }
    result_1: dict[str, Any] = agent.invoke(inputs_1, config)
    print(f"에이전트: {result_1['messages'][-1].content}")

    print("\n[턴 2] 사용자 입력: 'What was my name?'")
    inputs_2: dict[str, Any] = {
        "messages": [{"role": "user", "content": "What was my name?"}]
    }
    result_2: dict[str, Any] = agent.invoke(inputs_2, config)
    print(f"에이전트: {result_2['messages'][-1].content}")

    print("\n✅ 대화가 SQLite 데이터베이스에 저장되었습니다.")

    print("\n\n[재시작 시뮬레이션] 동일 thread_id로 새 대화를 시작합니다")
    print("-" * 80)

    print("\n[턴 3] 사용자 입력: 'Do you still remember my name?'")
    inputs_3: dict[str, Any] = {
        "messages": [{"role": "user", "content": "Do you still remember my name?"}]
    }
    # LangGraph는 SQLite에 축적된 상태를 읽어 동일 스레드의 컨텍스트를 재구성한다.
    result_3: dict[str, Any] = agent.invoke(inputs_3, config)
    print(f"에이전트: {result_3['messages'][-1].content}")

    print("\n✅ 이전 대화가 SQLite 데이터베이스에서 복원되었습니다.")

print("\n" + "=" * 80)
print("영속 메모리 기능 테스트를 완료했습니다!")
print("=" * 80)

if os.path.exists(db_path):
    file_size = os.path.getsize(db_path)
    print("\n💾 데이터베이스 파일 정보:")
    print(f"   - 경로: {os.path.abspath(db_path)}")
    print(f"   - 크기: {file_size:,} 바이트")
    print(
        "\n💡 이 파일을 삭제하지 않으면 프로그램을 다시 실행해도 대화 이력이 유지됩니다."
    )
