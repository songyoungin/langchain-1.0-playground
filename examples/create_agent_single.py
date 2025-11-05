from typing import Any

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# load the environment variables from the .env file
load_dotenv()


# define a tool function
def check_weather(location: str) -> str:
    """Return the weather forecast for the specified location.

    Args:
        location: The location to check the weather for.

    Returns:
        The weather forecast for the specified location.
    """
    return f"It's always sunny in {location}"


# create a Gemini model instance
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# create an agent with the create_agent function
agent: Any = create_agent(
    model=model,
    tools=[check_weather],
    system_prompt="You are a helpful assistant.",
)

# call the agent via invoke method
inputs: dict[str, Any] = {
    "messages": [{"role": "user", "content": "what is the weather in sf"}]
}
result: dict[str, Any] = agent.invoke(inputs)
print(f"Result: {result}")
