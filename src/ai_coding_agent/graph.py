import os
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from typing import Optional, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_core.prompts import (ChatPromptTemplate,MessagesPlaceholder)
from langchain_core.messages import ToolMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from ai_coding_agent.state import CodingState

from ai_coding_agent.tools import (
    write_file,
    read_file,
    list_files,
    execute,
    delete,
    shell,
    WORKSPACE,
)
from ai_coding_agent.tester import tester

# =====================================================
# LOAD ENVIRONMENT VARIABLES
# =====================================================

load_dotenv()


# =====================================================
# LLM & STRUCTURED OUTPUT MODELS
# =====================================================

# llm = ChatOllama(
#     model=os.getenv("OLLAMA_MODEL"),
#     temperature=0,
# )
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0,
)


class TestCase(BaseModel):
    """Structured test case specifying inputs and expected output."""

    input: str = Field(
        default="",
        description="Input data passed to the program (e.g. via stdin or function argument), or empty string if none."
    )
    expected_output: str = Field(
        description="Expected stdout output for this test case."
    )


class PlannerOutput(BaseModel):
    """Structured implementation plan, language routing, and test cases from the planner."""

    language: Literal["python", "cpp", "javascript", "bash", "other"] = Field(
        default="python",
        description="Target programming language for the solution (e.g. 'python', 'cpp', 'javascript', 'bash'). Default to 'python' unless the user specifies another language."
    )
    target_filename: str = Field(
        description="Suggested target filename with the correct extension (e.g. 'calculator.py', 'twosum.cpp', 'index.js')."
    )
    plan: str = Field(
        description="Step-by-step implementation plan breaking down requirements into clear logical steps without writing full code."
    )
    test_cases: list[TestCase] = Field(
        default_factory=list,
        description="List of structured test cases to verify the implementation. Each test case specifies an input and expected output."
    )


class RepairDiagnosis(BaseModel):
    """Structured diagnosis and repair plan for failed code execution."""

    failure_type: Literal[
        "compilation_error",
        "runtime_error",
        "output_mismatch",
        "missing_file",
        "timeout",
        "other",
    ] = Field(description="Category of the failure")
    root_cause: str = Field(
        description="Clear and concise explanation of what caused the failure"
    )
    suggested_fix: str = Field(
        description="Specific, actionable guidance on how the code should be modified to fix the issue"
    )


# Structured output LLMs
llm_with_structured_output = llm.with_structured_output(PlannerOutput)
llm_with_repair_diagnosis = llm.with_structured_output(RepairDiagnosis)


# =====================================================
# TOOLS & PERMISSION POLICY
# =====================================================

tools = [
    write_file,
    read_file,
    list_files,
    execute,
    delete,
    shell,
]

# Tool permissions:
# list_files → automatically execute
# read_file  → automatically execute
# write_file → optionally ask approval (controlled by ASK_APPROVAL_FOR_WRITE, default True)
# execute    → ask approval
# delete     → ask approval
# shell      → ask approval

ASK_APPROVAL_FOR_WRITE = os.getenv("ASK_APPROVAL_FOR_WRITE", "true").lower() in ("true", "1", "yes")

AUTO_APPROVED_TOOLS = {"list_files", "read_file"}
APPROVAL_REQUIRED_TOOLS = {"execute", "delete", "shell"}

if ASK_APPROVAL_FOR_WRITE:
    APPROVAL_REQUIRED_TOOLS.add("write_file")
else:
    AUTO_APPROVED_TOOLS.add("write_file")


# Give tools to the LLM
llm_with_tools = llm.bind_tools(tools)


# =====================================================
# PLANNER
# =====================================================
def extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


def planner(state: CodingState):

    print("\n--- PLANNER ---")

    prompt = [
        (
            "system",
            """You are a software development planner and language router.
Your job is to:
1. Identify the target programming language requested by the user (default to 'python' unless specified otherwise).
2. Suggest a suitable target filename with the correct file extension.
3. Create a simple step-by-step implementation plan without writing full code.
4. Formulate concrete test cases with input and expected_output to verify the implementation.""",
        ),
        (
            "human",
            f"The user wants:\n\n{state['user_request']}",
        ),
    ]

    result: PlannerOutput = llm_with_structured_output.invoke(prompt)

    plan = result.plan.strip()
    language = result.language or "python"
    target_filename = result.target_filename.strip()
    test_cases = [tc.model_dump() for tc in result.test_cases]

    # Combined expected output string for backwards compatibility
    expected_output = "\n".join(
        tc.get("expected_output", "") for tc in test_cases if tc.get("expected_output")
    )

    print("\nPLAN:")
    print(plan)
    print(f"Language: {language}")
    print(f"Target filename: {target_filename}")

    if test_cases:
        print("\nTEST CASES:")
        for idx, tc in enumerate(test_cases, 1):
            print(f"  [{idx}] Input: {tc.get('input', '')!r} -> Expected: {tc.get('expected_output', '')!r}")

    return {
        "plan": plan,
        "test_cases": test_cases,
        "expected_output": expected_output,
        "language": language,
        "filename": target_filename,
    }
# =====================================================
# CODER PROMPT
# =====================================================
coder_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an expert software developer working inside a workspace directory.

You have access to these tools:

1. list_files
   Use this to see what files exist (auto-approved).

2. read_file
   Use this to read an existing file (auto-approved).

3. write_file
   Use this to create or modify a file.

4. execute
   Use this to run a file in workspace and view its output.

5. delete
   Use this to delete a file.

6. shell
   Use this to run a shell command.

Rules:

- Target programming language: {language}. Write clean, valid code in this language.
- Target filename: {filename}.
- If you need to see available files, use list_files.
- If you need to inspect an existing file, use read_file.
- If you need to create or modify a file, use write_file.
- Never assume the contents of an existing file.
- Files must be inside the workspace.
- Include any example test cases or verification logic directly inside the target file (e.g. using `if __name__ == '__main__':` for Python), rather than creating separate subprocess wrapper scripts.
- If previous test failed or repair guidance is provided, inspect the diagnosis and fix the issue using write_file.
- Complete the user's request.
"""
        ),
        (
            "human",
            """
User request:
{user_request}

Target language:
{language}

Target filename:
{filename}

Implementation plan:
{plan}

Test cases to satisfy:
{test_cases_str}

Previous test result:
{test_result}

Repair guidance:
{repair_guidance}
"""
        ),
        MessagesPlaceholder(
            variable_name="messages",
            optional=True
        ),
    ]
)
# =====================================================
# CODER CHAIN
# =====================================================

coder_chain = coder_prompt | llm_with_tools


# =====================================================
# CODER NODE
# =====================================================
def coder(state: CodingState):

    iteration = state.get("iteration", 0) + 1

    print(f"\n--- CODER ITERATION {iteration} ---")

    test_cases_list = state.get("test_cases", [])
    if test_cases_list:
        test_cases_str = "\n".join(
            f"- Test Case {idx}: Input: {tc.get('input', '')!r} -> Expected Output: {tc.get('expected_output', '')!r}"
            for idx, tc in enumerate(test_cases_list, 1)
        )
    else:
        test_cases_str = state.get("expected_output", "None specified")

    response = coder_chain.invoke(
        {
            "user_request": state.get("user_request", ""),
            "language": state.get("language", "python"),
            "filename": state.get("filename", ""),
            "plan": state.get("plan", ""),
            "test_cases_str": test_cases_str,
            "test_result": state.get("test_result", ""),
            "repair_guidance": state.get("repair_guidance", "None"),
            "messages": state.get("messages", []),
        }
    )

    filename = state.get("filename", "")
    code = state.get("code", "")

    if response.tool_calls:

        for tool_call in response.tool_calls:

            if tool_call["name"] == "write_file":

                filename = tool_call["args"].get(
                    "filename",
                    filename
                )
                code = tool_call["args"].get(
                    "content",
                    code
                )

    # If the file exists on disk, read the actual file content
    if filename:
        file_path = WORKSPACE / filename
        if file_path.exists():
            try:
                code = file_path.read_text(encoding="utf-8")
            except Exception:
                pass

    return {
        "messages": [response],
        "code": code,
        "filename": filename,
        "iteration": iteration,
    }
# =====================================================
# HUMAN APPROVAL NODE
# =====================================================

def approval_node(state: CodingState):

    print("\n--- HUMAN APPROVAL ---")

    last_message = state["messages"][-1]

    # Filter for tool calls requiring approval
    pending_tool_calls = [
        tc for tc in last_message.tool_calls
        if tc["name"] in APPROVAL_REQUIRED_TOOLS
    ]

    if not pending_tool_calls:
        return Command(goto="tools")

    tool_call = pending_tool_calls[0]
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    approval = interrupt(
        {
            "message": f"The agent wants to execute a sensitive tool: {tool_name}",
            "tool": tool_name,
            "arguments": tool_args,
        }
    )

    if approval == "yes":

        print("\nTool approved.")

        return Command(
            goto="tools"
        )

    print("\nTool rejected.")

    return Command(
        goto="coder",
        update={
            "messages": [
                ToolMessage(
                    tool_call_id=tool_call["id"],
                    content=(
                        f"Human rejected the tool call: {tool_name}. "
                        "Do not use this tool call. "
                        "Choose another approach."
                    ),
                )
            ]
        },
    )
# =====================================================
# TOOL NODE
# =====================================================

tool_node = ToolNode(tools)


# =====================================================
# ROUTER
# =====================================================
def route_after_coder(state: CodingState):

    MAX_ITERATIONS = 5

    if state.get("iteration", 0) >= MAX_ITERATIONS:
        return "tester"

    last_message = state["messages"][-1]

    if last_message.tool_calls:
        # Check if any tool call requires approval
        requires_approval = any(
            tc["name"] in APPROVAL_REQUIRED_TOOLS
            for tc in last_message.tool_calls
        )
        if requires_approval:
            return "approval"
        # Auto-approved tools (e.g. list_files, read_file) execute immediately
        return "tools"

    if state.get("filename"):
        return "tester"

    return END

def route_after_tester(state: CodingState):

    test_result = state.get("test_result", "")

    if test_result.startswith("PASS"):
        return END

    MAX_REPAIRS = 3

    repair_attempts = state.get("repair_attempts", 0)

    if repair_attempts >= MAX_REPAIRS:
        print("\nMaximum repair attempts reached.")
        return END

    return "repair"

def repair_node(state: CodingState):

    repair_attempts = state.get("repair_attempts", 0) + 1

    print(
        f"\n--- REPAIR ATTEMPT {repair_attempts} ---"
    )

    filename = state.get("filename", "")
    if filename:
        file_path = WORKSPACE / filename
        if file_path.exists():
            try:
                code_content = file_path.read_text(encoding="utf-8")
            except Exception:
                code_content = state.get("code", "")
        else:
            code_content = state.get("code", "")
    else:
        code_content = state.get("code", "")

    test_cases_list = state.get("test_cases", [])
    if test_cases_list:
        test_cases_str = "\n".join(
            f"- Input: {tc.get('input', '')!r} -> Expected: {tc.get('expected_output', '')!r}"
            for tc in test_cases_list
        )
    else:
        test_cases_str = state.get("expected_output", "None")

    # Use structured output to diagnose the failure and formulate an actionable fix
    diagnosis_prompt = [
        (
            "system",
            """You are an expert software debugging and repair specialist.
Analyze the test failure output, code, and user request to provide a structured diagnosis and clear fix instructions."""
        ),
        (
            "human",
            f"""User request:
{state.get("user_request", "")}

Language:
{state.get("language", "python")}

Filename:
{filename}

Implementation plan:
{state.get("plan", "")}

Test cases:
{test_cases_str}

Code from disk:
{code_content}

Test result / error:
{state.get("test_result", "")}
"""
        )
    ]

    diagnosis: RepairDiagnosis = llm_with_repair_diagnosis.invoke(diagnosis_prompt)

    repair_guidance = (
        f"Failure Type: {diagnosis.failure_type}\n"
        f"Root Cause: {diagnosis.root_cause}\n"
        f"Suggested Fix: {diagnosis.suggested_fix}"
    )

    print(f"\n[REPAIR DIAGNOSIS]\n{repair_guidance}\n")

    return {
        "repair_attempts": repair_attempts,
        "repair_guidance": repair_guidance,
        "code": code_content,
        "messages": [
            HumanMessage(
                content=f"Test failed. Please apply this repair guidance:\n{repair_guidance}"
            )
        ]
    }

# =====================================================
# BUILD GRAPH
# =====================================================

builder = StateGraph(CodingState)


builder.add_node(
    "planner",
    planner
)

builder.add_node(
    "coder",
    coder
)

builder.add_node(
    "approval",
    approval_node
)

builder.add_node(
    "tools",
    tool_node
)

builder.add_node(
    "tester",
    tester
)

builder.add_node(
    "repair",
    repair_node
)
# START → PLANNER

builder.add_edge(
    START,
    "planner"
)


# PLANNER → CODER

builder.add_edge(
    "planner",
    "coder"
)


# CODER → APPROVAL or END

builder.add_conditional_edges(
    "coder",
    route_after_coder,
    {
        "approval": "approval",
        "tester": "tester",
        END: END,
    }
)

builder.add_conditional_edges(
    "tester",
    route_after_tester,
    {
        "repair": "repair",
        END: END,
    }
)

# TOOLS → CODER

builder.add_edge(
    "tools",
    "coder"
)

builder.add_edge(
    "repair",
    "coder"
)

# =====================================================
# COMPILE GRAPH
# =====================================================

memory = InMemorySaver()

graph = builder.compile(
    checkpointer=memory
)
