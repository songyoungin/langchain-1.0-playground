from typing import Any

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# .env 설정을 불러 LangGraph가 사용할 인증 정보를 준비한다.
load_dotenv()


def check_weather(location: str) -> str:
    """요청 장소의 날씨를 설명한다.

    Args:
        location: 날씨를 확인할 지역 이름.

    Returns:
        입력된 지역이 항상 맑다는 문장.
    """
    return f"It's always sunny in {location}"


model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# LangGraph create_agent API가 모델·도구를 연결해 단일 노드 그래프를 구성한다.
agent: Any = create_agent(
    model=model,
    tools=[check_weather],
    system_prompt="You are a helpful assistant.",
)

inputs: dict[str, Any] = {
    "messages": [{"role": "user", "content": "what is the weather in sf"}]
}
# LangGraph 런타임이 메시지를 받아 그래프 노드를 순차 실행해 결과를 반환한다.
result: dict[str, Any] = agent.invoke(inputs)
print(f"결과: {result}")
