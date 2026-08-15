"""OpenAI setup diagnostics for this project.

This utility checks whether ``OPENAI_API_KEY`` is available, verifies that the
configured model can answer a tiny request, and optionally lists accessible
models. It replaces the older separate API-key and model-list scripts.
"""

import argparse
import os

from dotenv import load_dotenv
from openai import OpenAI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check OpenAI API key access and list available models."
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model to test. Defaults to OPENAI_MODEL or gpt-5-mini.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List models available to the configured API key.",
    )
    parser.add_argument(
        "--check-completion",
        action="store_true",
        help="Send a tiny request to the selected model.",
    )
    return parser


def get_client() -> OpenAI | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OpenAI diagnostics failed: OPENAI_API_KEY is missing in .env")
        return None

    return OpenAI(api_key=api_key)


def list_models(client: OpenAI) -> bool:
    try:
        models = client.models.list()
    except Exception as error:
        print("Could not list OpenAI models.")
        print("Error type:", type(error).__name__)
        print("Error message:", error)
        return False

    print("Available models:")
    for model in models.data:
        print("-", model.id)
    return True


def check_completion(client: OpenAI, model: str) -> bool:
    try:
        response = client.responses.create(
            model=model,
            input="Reply with exactly: API diagnostics passed",
        )
    except Exception as error:
        print("OpenAI completion check failed.")
        print("Model:", model)
        print("Error type:", type(error).__name__)
        print("Error message:", error)
        return False

    print("OpenAI completion check successful.")
    print("Model:", model)
    print("Model reply:", response.output_text)
    return True


def main() -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()

    # If no specific action is selected, run both checks for a useful default.
    run_list_models = args.list_models or not args.check_completion
    run_completion_check = args.check_completion or not args.list_models
    model = args.model or os.getenv("OPENAI_MODEL", "gpt-5-mini")

    client = get_client()
    if client is None:
        return 1

    checks = []
    if run_list_models:
        checks.append(list_models(client))
    if run_completion_check:
        checks.append(check_completion(client, model))

    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())

