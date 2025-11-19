import os
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver

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

# SQLite database file path
db_path = "checkpoints.db"

print("=" * 80)
print("LangChain 1.0 - Persistent Memory Example (SqliteSaver)")
print("=" * 80)
print(f"\n💾 Database file: {os.path.abspath(db_path)}")
print()

# Create and use SqliteSaver checkpointer
with SqliteSaver.from_conn_string(db_path) as checkpointer:
    # Create agent with persistent memory using create_agent
    agent: Any = create_agent(
        model=model,
        tools=[save_user_name],
        checkpointer=checkpointer,  # Add SQLite checkpointer
        system_prompt="You are a helpful assistant that permanently remembers conversation history.",
    )

    # Conversation Session: thread_id = "persistent-user-1"
    print("[Session 1] thread_id: persistent-user-1")
    print("-" * 80)

    config: dict[str, Any] = {"configurable": {"thread_id": "persistent-user-1"}}

    # Turn 1: Introduce name
    print("\n[Turn 1] User: My name is Jane")
    inputs_1: dict[str, Any] = {
        "messages": [{"role": "user", "content": "My name is Jane"}]
    }
    result_1: dict[str, Any] = agent.invoke(inputs_1, config)
    print(f"Agent: {result_1['messages'][-1].content}")

    # Turn 2: Ask for name
    print("\n[Turn 2] User: What was my name?")
    inputs_2: dict[str, Any] = {
        "messages": [{"role": "user", "content": "What was my name?"}]
    }
    result_2: dict[str, Any] = agent.invoke(inputs_2, config)
    print(f"Agent: {result_2['messages'][-1].content}")

    print("\n✅ Conversation has been saved to SQLite database.")

    # Simulate restart (using same thread_id)
    print("\n\n[Restart Simulation] New conversation with same thread_id")
    print("-" * 80)

    # Turn 3: Ask for name again (loaded from DB)
    print("\n[Turn 3] User: Do you still remember my name?")
    inputs_3: dict[str, Any] = {
        "messages": [{"role": "user", "content": "Do you still remember my name?"}]
    }
    result_3: dict[str, Any] = agent.invoke(inputs_3, config)
    print(f"Agent: {result_3['messages'][-1].content}")

    print("\n✅ Previous conversation was loaded from SQLite database.")

print("\n" + "=" * 80)
print("Persistent memory feature test completed!")
print("=" * 80)

# Display database file info
if os.path.exists(db_path):
    file_size = os.path.getsize(db_path)
    print("\n💾 Database file information:")
    print(f"   - Path: {os.path.abspath(db_path)}")
    print(f"   - Size: {file_size:,} bytes")
    print(
        "\n💡 As long as this file is not deleted, conversation history persists across program restarts."
    )
