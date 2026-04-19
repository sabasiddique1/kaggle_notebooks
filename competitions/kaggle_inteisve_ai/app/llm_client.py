"""LLM client wrapper (Gemini or Mock)."""
import os
from typing import Optional
from app.observability import METRICS, log_event


class LLMClient:
    """Base LLM client interface."""
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class MockLLM(LLMClient):
    """Mock LLM for when API key is not available."""
    def generate(self, prompt: str) -> str:
        return (
            "I'm running in MOCK mode (no API key).\n"
            "Summary: Based on provided symptoms, prioritize safety. "
            "If severe symptoms are present (breathing trouble, chest pain, stroke signs, major bleeding, unconsciousness), seek emergency care immediately."
        )


class GeminiLLM(LLMClient):
    """Google Gemini LLM client."""
    def __init__(self, api_key: str):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def generate(self, prompt: str) -> str:
        METRICS["llm_calls"] += 1
        resp = self.model.generate_content(prompt)
        return resp.text


def get_llm_client() -> LLMClient:
    """Get LLM client (Gemini if key available, else Mock)."""
    api_key = os.environ.get("GEMINI_API_KEY")
    use_gemini = bool(api_key)
    
    if use_gemini:
        log_event("llm_ready", provider="gemini")
        return GeminiLLM(api_key)
    else:
        log_event("llm_ready", provider="mock")
        return MockLLM()






