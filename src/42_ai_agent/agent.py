import os
import sys
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# 1. Load API Key
load_dotenv()
api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    sys.exit("❌ Error: OPENROUTER_API_KEY is not set in your .env file.")

# 2. Define Deterministic Tools
@tool
def calculate(expression: str) -> str:
    """Safely evaluates basic arithmetic expressions like '42 * 1024' or '(100 - 25) / 5'."""
    try:
        allowed = set("0123456789+-*/(). %")
        if not set(expression).issubset(allowed):
            return "Error: Expression contains forbidden characters."
        return str(eval(expression, {"__builtins__": None}, {}))
    except Exception as e:
        return f"Calculation error: {e}"

@tool
def get_system_info() -> str:
    """Returns local operating system details and active directory path."""
    import platform
    return f"OS: {platform.system()} {platform.release()}, Current Path: {os.getcwd()}"

@tool
def list_directory(path: str = ".") -> str:
    """Lists files and folders in a given directory path."""
    try:
        items = os.listdir(path)
        return "\n".join(items) if items else "Directory is empty."
    except Exception as e:
        return f"Error listing directory: {e}"


@tool
def read_file_preview(filepath: str, max_chars: int = 500) -> str:
    """Reads the first N characters of a text file to prevent context window overload."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read(max_chars)
        return f"--- Content of {filepath} (first {max_chars} chars) ---\n{content}"
    except Exception as e:
        return f"Error reading file: {e}"


@tool
def save_scratchpad_note(filename: str, content: str) -> str:
    """Writes or appends an important summary or finding to an external text file."""
    try:
        # Enforce writing to a safe .txt file only
        if not filename.endswith(".txt"):
            filename += ".txt"
        with open(filename, "a", encoding="utf-8") as f:
            f.write(content + "\n")
        return f"Successfully saved note to {filename}."
    except Exception as e:
        return f"Error writing note: {e}"


@tool
def flaky_network_fetch(endpoint: str) -> str:
    """Simulates an unreliable external API that always fails with an ambiguous error."""
    return "HTTP 500: Internal Server Error. Syntax rejected, try another format."


# Update tool registry:
tools = [calculate, get_system_info, flaky_network_fetch]

# 3. Configure Model with OpenRouter API
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    model="poolside/laguna-s-2.1:free", # Reliable free model for tool-calling
    temperature=0,
    default_headers={
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "42 Berlin AI Workshop",
    }
)

# 4. Compile ReAct Graph with Checkpointer Memory
memory = MemorySaver()
agent = create_react_agent(
    model=llm,
    tools=tools,
    checkpointer=memory,
    prompt="You are a helpful AI assistant with access to local tools. Always verify computations and system details using your tools before answering."
)

# 5. Interactive Chat REPL Loop
def run_interactive_session():
    config = {"configurable": {"thread_id": "session_42_berlin"}}
    print("=" * 60)
    print("🤖 Agent Ready! (Connected to OpenRouter Free Tier)")
    print("Type your message, or type 'exit' / 'q' to quit.")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYou > ").strip()
            if user_input.lower() in ("exit", "quit", "q"):
                print("👋 Exiting workshop session...")
                break
            if not user_input:
                continue

            print("\n⚙️ Processing (Streaming Graph Steps)...")
            # Stream and inspect intermediate message steps
            for event in agent.stream(
                {"messages": [("user", user_input)]},
                config=config,
                stream_mode="values"
            ):
                latest = event["messages"][-1]
                # Only print new agent responses and tool executions
                if latest.type in ("ai", "tool") and latest.content != user_input:
                    latest.pretty_print()

        except KeyboardInterrupt:
            print("\nSession stopped.")
            break
        except Exception as err:
            print(f"⚠️ Error: {err}")


if __name__ == "__main__":
    run_interactive_session()
