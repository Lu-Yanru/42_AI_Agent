import time
from agent import agent

# Define benchmark tasks with deterministic ground truth assertions
EVAL_SUITE = [
    {
        "id": "eval_math",
        "input": "Calculate 1024 * 64 exactly.",
        "expected_tool": "calculate",
        "expected_substring": "65,536"
    },
    {
        "id": "eval_system",
        "input": "What OS am I running?",
        "expected_tool": "get_system_info",
        "expected_substring": "Linux"  # Or "Windows" / "Linux" depending on the laptop
    },
    {
        "id": "eval_pure_reasoning",
        "input": "What is the capital of France? Answer in one word.",
        "expected_tool": None,  # Should NOT invoke tools
        "expected_substring": "Paris"
    }
]

def run_evals():
    print("================ RUNNING AGENT EVALUATION SUITE ================")
    passed = 0

    for test in EVAL_SUITE:
        print(f"\n[Test ID]: {test['id']}")
        print(f" Query: {test['input']}")

        tools_called = []
        final_answer = ""
        config = {"configurable": {"thread_id": f"eval_{test['id']}_{time.time()}"}}

        # Execute agent stream
        for event in agent.stream(
            {"messages": [("user", test["input"])]}, 
            config=config, 
            stream_mode="values"
        ):
            msg = event["messages"][-1]
            if msg.type == "ai" and msg.tool_calls:
                for call in msg.tool_calls:
                    tools_called.append(call["name"])
            elif msg.type == "ai" and not msg.tool_calls:
                final_answer = msg.content

        # 1. Deterministic Tool Assertion
        tool_passed = True
        if test["expected_tool"]:
            tool_passed = test["expected_tool"] in tools_called
        else:
            tool_passed = len(tools_called) == 0

        # 2. Output Fact Assertion
        content_passed = test["expected_substring"].lower() in final_answer.lower()

        test_success = tool_passed and content_passed
        if test_success:
            passed += 1
            print(f" -> Result: PASS ✅ (Tools: {tools_called})")
        else:
            print(f" -> Result: FAIL ❌ (Tools: {tools_called}, Expected Substring: '{test['expected_substring']}')")
            print(f"    Agent Output: {final_answer}")

    success_rate = (passed / len(EVAL_SUITE)) * 100
    print("\n================ EVALUATION SUMMARY ================")
    print(f"Total Passed: {passed}/{len(EVAL_SUITE)} ({success_rate:.1f}%)")

if __name__ == "__main__":
    run_evals()
