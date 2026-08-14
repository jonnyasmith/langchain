from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.base import OpenAIRefusalError

from extractor.chain import extract
from extractor.schemas import SCHEMAS

DEFAULT_MODEL = "gpt-5-nano"
MAX_DOCUMENT_CHARACTERS = 100_000


class ConfigurationError(Exception):
    """The extractor cannot construct its provider model."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract typed data from one source document.")
    parser.add_argument("input", nargs="?", help="source file path, or - to read stdin")
    parser.add_argument("--schema", help="named extraction schema")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model id")
    parser.add_argument("--list-schemas", action="store_true", help="list named schemas")
    parser.add_argument(
        "--debug", action="store_true", help="write the raw model message to stderr"
    )
    return parser


def _real_model(model_id: str) -> BaseChatModel:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise ConfigurationError("missing OPENAI_API_KEY; set it in extractor/.env")
    return ChatOpenAI(model=model_id, reasoning_effort="none", temperature=0)


def main(
    argv: Sequence[str] | None = None,
    *,
    stdin: TextIO = sys.stdin,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
    model: BaseChatModel | None = None,
) -> int:
    args = _parser().parse_args(argv)
    if args.list_schemas:
        stdout.write("\n".join(sorted(SCHEMAS)) + "\n")
        return 0
    if args.schema is None or args.input is None:
        stderr.write("Input error: --schema and an input path are required.\n")
        return 1
    schema = SCHEMAS.get(args.schema)
    if schema is None:
        valid_names = ", ".join(sorted(SCHEMAS))
        stderr.write(f"Unknown schema {args.schema!r}. Valid schemas: {valid_names}.\n")
        return 1
    try:
        document = (
            stdin.read() if args.input == "-" else Path(args.input).read_text(encoding="utf-8")
        )
    except OSError as error:
        stderr.write(f"Input file error for {args.input!r}: {error.strerror or error}.\n")
        return 1
    document_size = len(document)
    if document_size > MAX_DOCUMENT_CHARACTERS:
        stderr.write(
            "Document too large: "
            f"maximum is {MAX_DOCUMENT_CHARACTERS:,} characters; "
            f"received {document_size:,} characters.\n"
        )
        return 1
    try:
        result = extract(document, schema, model or _real_model(args.model))
    except OpenAIRefusalError as error:
        stderr.write(f"Refusal: {error}\n")
        return 4
    except ConfigurationError as error:
        stderr.write(f"Configuration error: {error}\n")
        return 1
    except Exception as error:
        stderr.write(f"Unexpected error: {error}\n")
        return 1
    if args.debug:
        stderr.write(f"Raw model message: {result['raw']!r}\n")
    parsing_error = result["parsing_error"]
    if isinstance(parsing_error, OpenAIRefusalError):
        stderr.write(f"Refusal: {parsing_error}\n")
        return 4
    if parsing_error is not None:
        stderr.write(f"Validation failure: {parsing_error}\n")
        return 2
    parsed = result["parsed"]
    if parsed is None:
        stderr.write("Empty extraction: the model returned no object.\n")
        return 3
    stdout.write(parsed.model_dump_json() + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
