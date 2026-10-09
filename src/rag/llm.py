import ollama

DEFAULT_MODEL = "llama3.1:8b"


class LLMError(Exception):
    """Raised when the LLM backend cannot produce an answer."""


class OllamaLLM:
    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.model = model

    def generate(self, system: str, user: str) -> str:
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                options={"temperature": 0},
            )
        except Exception as exc:
            raise LLMError(f"Ollama call failed: {exc}") from exc
        return str(response["message"]["content"])
