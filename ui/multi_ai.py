# multi_ai.py  –  Single OpenRouter provider
import os
import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# Popular models available on OpenRouter
OPENROUTER_MODELS = [
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3-haiku",
    "google/gemini-2.0-flash-exp:free",
    "google/gemini-flash-1.5",
    "meta-llama/llama-3.3-70b-instruct:free",
    "meta-llama/llama-3.1-8b-instruct:free",
    "mistralai/mixtral-8x7b-instruct",
    "deepseek/deepseek-chat",
    "qwen/qwen-2.5-72b-instruct",
]


class MultiAIProvider:
    """
    Single OpenRouter provider that exposes 100+ models
    via one unified API key.
    """

    def __init__(self, model: str = None, api_key: str = None):
        self.model = model or OPENROUTER_MODELS[6]  # free llama default
        self.api_key = (
            api_key
            or os.getenv("OPENROUTER_API_KEY", "")
        )

    def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        if not self.api_key:
            return (
                "❌ No OpenRouter API key set. "
                "Add it in the sidebar Settings panel."
            )
        try:
            resp = requests.post(
                f"{OPENROUTER_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://db-intelligence-agent",
                    "X-Title": "DB Intelligence Agent",
                },
                json={
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                },
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.HTTPError as e:
            return f"❌ OpenRouter HTTP error: {e.response.status_code} — {e.response.text[:200]}"
        except Exception as e:
            return f"❌ OpenRouter Error: {str(e)}"

    @staticmethod
    def available_models():
        return OPENROUTER_MODELS

    @staticmethod
    def check_key(api_key: str = None) -> bool:
        key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        return bool(key and key.startswith("sk-or-"))
