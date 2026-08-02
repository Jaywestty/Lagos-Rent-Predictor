import os

from dotenv import load_dotenv

load_dotenv()


def get_env(name: str, default: str = None, required: bool = False) -> str:
    value = os.getenv(name, default)
    if required and not value:
        raise RuntimeError(f"Required environment variable '{name}' is not set")
    return value


GROQ_API_KEY = get_env("GROQ_API_KEY")
TAVILY_API_KEY = get_env("TAVILY_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"