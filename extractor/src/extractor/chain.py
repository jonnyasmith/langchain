from typing import Any, TypedDict, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel


class ExtractionResult(TypedDict):
    raw: BaseMessage
    parsed: BaseModel | None
    parsing_error: BaseException | None


def extract(
    document: str,
    schema: type[BaseModel],
    model: BaseChatModel,
) -> ExtractionResult:
    """Extract one validated object and retain the raw provider response."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Extract only facts stated in the source document. "
                "Do not infer or guess; use null when the source does not answer a field.",
            ),
            ("human", "{document}"),
        ]
    )
    structured_model = model.with_structured_output(
        schema,
        method="json_schema",
        strict=True,
        include_raw=True,
    )
    chain = prompt | structured_model
    result = cast(dict[str, Any], chain.invoke({"document": document}))
    return ExtractionResult(
        raw=result["raw"],
        parsed=result["parsed"],
        parsing_error=result["parsing_error"],
    )
