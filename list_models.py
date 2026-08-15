"""Utility script for listing OpenAI models available to the configured API key."""

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

models = client.models.list()

for model in models.data:
    print(model.id)
