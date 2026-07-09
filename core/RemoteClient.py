import time
import requests

from data.Response import Response
from data.Request import Request, Category


class RemoteClient:
    def __init__(self, api_key, base_url, allowed_models):
        self.api_key = api_key
        self.base_url = (base_url or "").rstrip("/")
        self.allowed_models = allowed_models or []

    def generate(
        self,
        task_data: Request,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 160
    ) -> Response:
        response = Response()

        if not self.api_key or not self.base_url:
            response.success = False
            response.error = "Missing Fireworks API configuration"
            response.route = "remote_error"
            return response

        if model not in self.allowed_models:
            response.success = False
            response.error = f"Model not allowed: {model}"
            response.route = "remote_error"
            return response

        prompt = self._build_prompt(task_data)

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a concise English assistant. "
                        "Answer exactly as requested. "
                        "Do not add unnecessary explanation."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }

        last_error = None

        for attempt in range(2):  # one retry
            try:
                api_response = requests.post(
                    self._chat_url(),
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json=payload,
                    timeout=25
                )

                api_response.raise_for_status()
                data = api_response.json()

                answer = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )

                cleaned_answer = self._clean_answer(answer)

                if not cleaned_answer:
                    response.success = False
                    response.error = "Remote model returned empty answer"
                    response.route = "remote_error"
                    response.model = model
                    response.raw_response = data
                    response.usage = data.get("usage", {})
                    return response

                response.success = True
                response.set_text(cleaned_answer)
                response.raw_response = data
                response.usage = data.get("usage", {})
                response.model = model
                response.route = "remote"

                return response

            except Exception as error:
                last_error = str(error)
                time.sleep(0.5)

        response.success = False
        response.error = f"Remote call failed: {last_error}"
        response.model = model
        response.route = "remote_error"
        return response

    def _chat_url(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url

        return f"{self.base_url}/chat/completions"

    def _build_prompt(self, task: Request) -> str:
        category = task.category
        prompt = task.prompt

        if category == Category.SENTIMENT:
            return (
                "Classify sentiment. If the user requested a specific format, follow it. "
                "Otherwise return: Label — brief reason. "
                "Valid labels: Positive, Negative, Neutral, Mixed.\n\n"
                f"Text:\n{prompt}"
            )

        if category == Category.NER:
            return (
                "Extract named entities. If the user requested specific entity types or format, follow that. "
                "Otherwise return JSON with keys: person, organization, location, date. "
                "Use empty arrays when none.\n\n"
                f"Text:\n{prompt}"
            )

        if category == Category.SUMMARISATION:
            return (
                "Summarize exactly as requested. Respect any sentence, word, bullet, or length constraint. "
                "Be concise.\n\n"
                f"Text:\n{prompt}"
            )

        if category == Category.MATH:
            return (
                "Solve the problem carefully. Return the final answer with a brief calculation only.\n\n"
                f"Problem:\n{prompt}"
            )

        if category == Category.LOGICAL:
            return (
                "Solve all constraints. Return the final answer and a brief justification only.\n\n"
                f"Problem:\n{prompt}"
            )

        if category == Category.CODE_DEBUG:
            return (
                "Fix the code. Return corrected code only unless the prompt asks for explanation.\n\n"
                f"Prompt:\n{prompt}"
            )

        if category == Category.CODE_GENERATION:
            return (
                "Write the requested function or code. Return clean code only unless the prompt asks for explanation.\n\n"
                f"Spec:\n{prompt}"
            )

        if category == Category.FACTUAL_KNOWLEDGE:
            return (
                "Answer accurately and concisely in English.\n\n"
                f"Question:\n{prompt}"
            )

        return prompt

    def _clean_answer(self, answer: str) -> str:
        if answer is None:
            return ""

        cleaned = answer.strip()

        if cleaned.startswith("```") and cleaned.endswith("```"):
            lines = cleaned.splitlines()
            if len(lines) >= 3:
                cleaned = "\n".join(lines[1:-1]).strip()

        return cleaned.strip()