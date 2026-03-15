# multi_ai.py
import os
from dotenv import load_dotenv

load_dotenv()

class MultiAIProvider:
    """
    Supports:
    ✅ Anthropic  (Claude)
    ✅ OpenAI     (GPT-4o)
    ✅ Google     (Gemini)
    ✅ Groq       (LLaMA - Free & Fast)
    """

    PROVIDERS = {
        "anthropic": {
            "name": "Claude (Anthropic)",
            "icon": "🟠",
            "models": [
                "claude-sonnet-4-20250514",
                "claude-opus-4-20250514"
            ],
            "env_key": "ANTHROPIC_API_KEY"
        },
        "openai": {
            "name": "GPT (OpenAI)",
            "icon": "🟢",
            "models": [
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4-turbo"
            ],
            "env_key": "OPENAI_API_KEY"
        },
        "gemini": {
            "name": "Gemini (Google)",
            "icon": "🔵",
            "models": [
                "gemini-2.0-flash",
                "gemini-1.5-pro",
                "gemini-1.5-flash"
            ],
            "env_key": "GOOGLE_API_KEY"
        },
        "groq": {
            "name": "Groq ⚡ (Free & Fast)",
            "icon": "🟣",
            "models": [
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "mixtral-8x7b-32768",
                "gemma2-9b-it"
            ],
            "env_key": "GROQ_API_KEY"
        }
    }

    def __init__(self, provider: str = "anthropic", model: str = None):
        self.provider = provider.lower()
        self.model = model or self.PROVIDERS[self.provider]["models"][0]
        self.client = None
        self._init_client()

    def _init_client(self):
        if self.provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )

        elif self.provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY")
            )

        elif self.provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            self.client = genai.GenerativeModel(self.model)

        elif self.provider == "groq":
            from groq import Groq
            self.client = Groq(
                api_key=os.getenv("GROQ_API_KEY")
            )

    def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        """
        One function → works for ALL providers
        """
        try:
            # ── Anthropic ──────────────────────────────
            if self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text

            # ── OpenAI ─────────────────────────────────
            elif self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content

            # ── Gemini ─────────────────────────────────
            elif self.provider == "gemini":
                response = self.client.generate_content(prompt)
                return response.text

            # ── Groq ───────────────────────────────────
            elif self.provider == "groq":
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content

        except Exception as e:
            return f"❌ AI Error ({self.provider} / {self.model}): {str(e)}"

    @classmethod
    def get_available(cls) -> dict:
        """Check which providers have API keys set"""
        available = {}
        for key, info in cls.PROVIDERS.items():
            if os.getenv(info["env_key"]):
                available[key] = info
        return available

    @classmethod
    def check_key(cls, provider: str) -> bool:
        env_key = cls.PROVIDERS[provider]["env_key"]
        return bool(os.getenv(env_key))
