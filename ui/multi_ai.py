# multi_ai.py  –  Multi-provider: OpenAI ChatGPT + Google Gemini
import os
from dotenv import load_dotenv

load_dotenv()

# ── OpenAI ChatGPT models ──────────────────────────────────────
OPENAI_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
]

# ── Google Gemini models ───────────────────────────────────────
GOOGLE_MODELS = [
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]

ALL_MODELS = OPENAI_MODELS + GOOGLE_MODELS

DEFAULT_MODEL = "gpt-4o-mini"


def _provider_of(model: str) -> str:
    """Return 'openai' or 'google' based on model name."""
    if model in GOOGLE_MODELS:
        return "google"
    return "openai"


class MultiAIProvider:
    """
    Multi-provider wrapper supporting OpenAI ChatGPT and Google Gemini
    via their respective official SDKs.
    """

    def __init__(self, model: str = None, api_key: str = None):
        self.model = model or os.getenv("DEFAULT_AI_MODEL", DEFAULT_MODEL)
        self.provider = _provider_of(self.model)

        if self.provider == "google":
            self.api_key = (
                api_key
                or os.getenv("GOOGLE_API_KEY", "")
            )
        else:
            self.api_key = (
                api_key
                or os.getenv("OPENAI_API_KEY", "")
            )

    def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        if not self.api_key:
            return (
                f"❌ No API key set for {self.provider.title()}. "
                "Add it in the sidebar Settings panel."
            )

        if self.provider == "openai":
            return self._call_openai(prompt, max_tokens)
        else:
            return self._call_google(prompt, max_tokens)

    def _call_openai(self, prompt: str, max_tokens: int) -> str:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            resp = client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
                timeout=60,
            )
            return resp.choices[0].message.content
        except ImportError:
            return "❌ openai package not installed. Run: pip install openai"
        except Exception as e:
            return f"❌ OpenAI Error: {str(e)}"

    def _call_google(self, prompt: str, max_tokens: int) -> str:
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model)
            resp = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                ),
            )
            return resp.text
        except ImportError:
            return (
                "❌ google-generativeai package not installed. "
                "Run: pip install google-generativeai"
            )
        except Exception as e:
            return f"❌ Google Gemini Error: {str(e)}"

    @staticmethod
    def available_models():
        return ALL_MODELS

    @staticmethod
    def openai_models():
        return OPENAI_MODELS

    @staticmethod
    def google_models():
        return GOOGLE_MODELS

    @staticmethod
    def check_openai_key(api_key: str = None) -> bool:
        key = api_key or os.getenv("OPENAI_API_KEY", "")
        return bool(key and key.startswith("sk-"))

    @staticmethod
    def check_google_key(api_key: str = None) -> bool:
        key = api_key or os.getenv("GOOGLE_API_KEY", "")
        return bool(key)
