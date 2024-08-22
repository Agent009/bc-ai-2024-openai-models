import os
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialise the OpenAI client
print(f'Using OPENAI_API_KEY, {OPENAI_API_KEY}')
client = OpenAI(api_key=OPENAI_API_KEY)
models = [
    "gpt-3.5-turbo",        # Up to Sep 2021, most cost-efficient model, performs relatively well on a variety of tasks
                            # Fine-tuning can improve its performance
                            # Cost for input is $0.50 per 1M tokens
    "gpt-4",                # Up to Sep 2021, human-level performance on various professional and academic benchmarks
                            # More reliable, creative, and able to handle much more nuanced instructions than GPT-3.5
                            # Cost for input is $30 per 1M tokens
    "gpt-4-0125-preview"]   # Up to Dec 2023, latest version of GPT-4 (as of March 2024) intended to reduce cases of
                            # “laziness” where the model doesn’t complete a task
                            # Cost for input is $10 per 1M tokens

def test_models():
    messages = [
        {
            "role": "user",
            "content": """
                Sarah has 5 brothers. Each of Sarah's brothers has 2 sisters. How many sisters does Sarah have in total?
                """,
        }
    ]

    for model in models:
        print(f"\n---\nGenerating chat completion with {model}:\n")
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )

        for chunk in stream:
            print(chunk.choices[0].delta.content or "", end="")


if __name__ == '__main__':
    test_models()

