"""
LangChain Core Concepts - LCEL and Runnables
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()


def demo_basic_chain():
    """Demonstrates a basic chain using LCEL and Runnables."""

    # Component 1: Define the prompt template using LCEL
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Answer in one sentence: {question}"
    )
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    parser = StrOutputParser()

    # Compose with pipe operator
    # chain = prompt | model
    chain = prompt | model | parser

    # Execute the chain with an input
    result = chain.invoke({"question": "What is LangChain?"})
    print(f"Response: {result}")

    return chain


def demo_batch_exectution():
    """Demonstrate batch execution for multiple inputs."""
    prompt = ChatPromptTemplate.from_template("Translate to Hinglish: {text}")
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    parser = StrOutputParser()

    chain = prompt | model | parser

    # Batch - run with multiple inputs
    inputs = [
        {"text": "Hello, how are you?"},
        {"text": "What is your name?"},
        {"text": "Where is the nearest restaurant?"},
    ]
    results = chain.batch(inputs)

    for text in zip(inputs, results):
        print(f"Input: {text[0]['text']} => Output: {text[1]}")


def demo_streaming():
    """Demonstrate streaming for real-time output."""
    # prompt = ChatPromptTemplate.from_template("Write a haiku about: {topic}")
    prompt = ChatPromptTemplate.from_template("Write a 10 lines about: {topic}")
    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
    )
    parser = StrOutputParser()

    chain = prompt | model | parser

    # Streaming - run with streaming enabled
    print("Streaming output: ")
    # for chunk in chain.stream({"topic": "nature"}):
    for chunk in chain.stream({"topic": "school"}):
        print(chunk, end="", flush=True)
    print()  # for newline after streaming


def demo_schema_inspection():
    """Demonstrate input/output schema inspection."""
    prompt = ChatPromptTemplate.from_template("Summarize the following text: {text}")
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    parser = StrOutputParser()

    chain = prompt | model | parser

    # Inspect input and output schemas
    input_schema = chain.input_schema.model_json_schema()
    output_schema = chain.output_schema.model_json_schema()

    print(f"Input Schema: {input_schema}")
    print(f"Output Schema: {output_schema}")


def exercise_first_chain():
    """Exercise the first chain."""
    my_prompt = "You are a marketing content writer. Generate a marketing tagline for the following product: {product} and audience: {audience}. And return just the tagline as a string"
    prompt = ChatPromptTemplate.from_template(my_prompt)
    # model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    # model = init_chat_model("gpt-4o-mini", temperature=0.7, max_tokens=1500)
    # model = init_chat_model(
    #     "openai:openai/gpt-oss-20b",
    #     temperature=0,
    #     api_key=os.getenv("OPENROUTER_API_KEY"),
    #     base_url="https://openrouter.ai/api/v1",
    # )
    model = init_chat_model(
        "openai/gpt-oss-20b",
        model_provider="openrouter",
        temperature=0.7,
    )
    parser = StrOutputParser()

    chain = prompt | model | parser

    result = chain.invoke({"product": "Silemile", "audience": "Gen Z"})
    print(f"Result: {result}")

    return chain


if __name__ == "__main__":
    # demo_basic_chain()
    # demo_batch_exectution()
    # demo_streaming()
    # demo_schema_inspection()
    exercise_first_chain()
