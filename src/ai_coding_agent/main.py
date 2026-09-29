from langgraph.types import Command
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ai_coding_agent.graph import graph


config = {
    "configurable": {
        "thread_id": "user-1"
    }
}


while True:

    user_input = input("\nYou: ")

    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    initial_state = {
        "user_request": user_input,
        "plan": "",
        "code": "",
        "filename": "",
        "language": "",
        "test_cases": [],
        "repair_guidance": "",
        "expected_output": "",
        "test_result": "",
        "iteration": 0,
        "repair_attempts": 0,
        "messages": [],
    }

    result = graph.invoke(
        initial_state,
        config=config
    )

    # -----------------------------------------------
    # GRAPH WAS INTERRUPTED
    # -----------------------------------------------

    while "__interrupt__" in result:

        interrupt_info = result["__interrupt__"][0].value

        print("\n⚠️ HUMAN APPROVAL REQUIRED")

        print("Tool:", interrupt_info["tool"])

        print("Arguments:", interrupt_info["arguments"])

        answer = input("\nAllow this tool? (yes/no): ")

        result = graph.invoke(
            Command(resume=answer.lower()),
            config=config
        )

    # -----------------------------------------------
    # FINAL RESULT
    # -----------------------------------------------

    print("\n--- FINAL RESULT ---")

    if result.get("test_result"):

        print(result["test_result"])

        print(
            f"File: {result.get('filename', 'Unknown')}"
        )

        print(
            f"Repair attempts: "
            f"{result.get('repair_attempts', 0)}"
        )

    elif result.get("code"):

        print(result["code"])

    else:

        print("Request completed.")
