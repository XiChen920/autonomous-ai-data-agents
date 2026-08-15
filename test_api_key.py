"""Small health-check script for verifying the OpenAI API key and model access."""

import os

from dotenv import load_dotenv
from openai import OpenAI


def main():
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("API test failed: OPENAI_API_KEY is missing in .env")
        return

    client = OpenAI(api_key=api_key)

    try:
        response = client.responses.create(
            model="gpt-5-mini",
            input="Reply with exactly: API test passed",
        )

        print("API test successful.")
        print("Model reply:", response.output_text)

    except Exception as error:
        print("API test failed.")
        print("Error type:", type(error).__name__)
        print("Error message:", error)


if __name__ == "__main__":
    main()
