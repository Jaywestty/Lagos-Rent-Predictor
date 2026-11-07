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
GROQ_MODEL = get_env("GROQ_MODEL", "openai/gpt-oss-120b")

LOW_DATA_MAX = int(os.getenv("LOW_DATA_MAX", "3"))
LIMITED_CONFIDENCE_MAX = int(os.getenv("LIMITED_CONFIDENCE_MAX", "9"))
PERCENTILE_MIN_COUNT = int(os.getenv("PERCENTILE_MIN_COUNT", "10"))
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "30"))
