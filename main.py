import os
from importlib.metadata import version

from dotenv import load_dotenv

load_dotenv()

core_version = version("langchain-core")
lg_version = version("langgraph")
from langchain_openai import ChatOpenAI

print(f"langchain-core version: {core_version}")
print(f"langgraph version: {lg_version}")


def main():

    # Test openai
    # llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    llm = ChatOpenAI(
        temperature=0,
        model="openai/gpt-oss-20b",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )
    response = llm.invoke("Say 'setup complete!' in one word")
    print(f"Response from ChatOpenAI: {response}")

    # Test anthropic
    # llm_anthropic = ChatOpenAI(model="claude-sonnet-4-5-20250929", temperature=0)
    llm_anthropic = ChatOpenAI(
        temperature=0,
        model="anthropic/claude-sonnet-4.5",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )
    response_anthropic = llm_anthropic.invoke("Say 'setup complete!' in one word")
    print(f"Response from ChatAnthropic: {response_anthropic}")

    print("Setup complete!")


if __name__ == "__main__":
    main()
