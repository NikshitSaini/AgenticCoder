from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph.message import add_messages

from langchain_core.messages import BaseMessage


class CodingState(TypedDict, total=False):

    user_request: str

    plan: str

    code: str

    filename: str

    language: str

    expected_output: str

    test_cases: list[dict]

    test_result: str

    repair_guidance: str

    iteration: int

    repair_attempts: int

    messages: Annotated[
        list[BaseMessage],
        add_messages
    ]
