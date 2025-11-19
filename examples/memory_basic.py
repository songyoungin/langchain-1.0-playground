from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables
load_dotenv()


# Define tool function
def save_user_name(name: str) -> str:
    """Save the user's name.

    Args:
        name: The user's name to save.

    Returns:
        A confirmation message that the name was saved.
    """
    return f"I've remembered your name, {name}."


# Create Gemini model instance
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Create MemorySaver checkpointer (in-memory storage)
memory = MemorySaver()

# Create agent with memory using create_agent
agent: Any = create_agent(
    model=model,
    tools=[save_user_name],
    checkpointer=memory,  # Add memory checkpointer
    system_prompt="You are a helpful assistant that remembers conversation history.",
)

print("=" * 80)
print("LangChain 1.0 - Basic Memory Example (MemorySaver)")
print("=" * 80)
print()

# Conversation Session 1: thread_id = "user-1"
print("[Session 1] thread_id: user-1")
print("-" * 80)

config_1: dict[str, Any] = {"configurable": {"thread_id": "user-1"}}

# Turn 1: Introduce name
print("\n[Turn 1] User: My name is John")
inputs_1: dict[str, Any] = {
    "messages": [{"role": "user", "content": "My name is John"}]
}
result_1: dict[str, Any] = agent.invoke(inputs_1, config_1)
print(f"Agent: {result_1['messages'][-1].content}")

# Turn 2: Ask for name
print("\n[Turn 2] User: What was my name?")
inputs_2: dict[str, Any] = {
    "messages": [{"role": "user", "content": "What was my name?"}]
}
result_2: dict[str, Any] = agent.invoke(inputs_2, config_1)
print(f"Agent: {result_2['messages'][-1].content}")

# Conversation Session 2: thread_id = "user-2" (different session)
print("\n\n[Session 2] thread_id: user-2 (new session)")
print("-" * 80)

config_2: dict[str, Any] = {"configurable": {"thread_id": "user-2"}}

# Turn 1: Ask for name (no previous session info)
print("\n[Turn 1] User: What was my name?")
inputs_3: dict[str, Any] = {
    "messages": [{"role": "user", "content": "What was my name?"}]
}
result_3: dict[str, Any] = agent.invoke(inputs_3, config_2)
print(f"Agent: {result_3['messages'][-1].content}")
print("\n💡 Sessions are isolated, so it doesn't remember the previous conversation.")

# Return to Session 1
print("\n\n[Return to Session 1] thread_id: user-1")
print("-" * 80)

# Turn 3: Ask for name again
print("\n[Turn 3] User: Please tell me my name again")
inputs_4: dict[str, Any] = {
    "messages": [{"role": "user", "content": "Please tell me my name again"}]
}
result_4: dict[str, Any] = agent.invoke(inputs_4, config_1)
print(f"Agent: {result_4['messages'][-1].content}")
print("\n✅ Session 1's conversation history is still maintained.")

print("\n" + "=" * 80)
print("Memory feature test completed!")
print("=" * 80)
print(
    "\n💡 Note: MemorySaver is in-memory storage. Data is lost when the program exits."
)
