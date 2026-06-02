from dotenv import load_dotenv
from langchain_ollama import ChatOllama


def get_planner_llm():
    
    return ChatOllama(
        model="qwen3:4b-instruct",
        temperature=0,
    )

