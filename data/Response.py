class Response:
    def __init__(self):
        self.success = False
        self.response = ""
        self.answer = ""
        self.model = None
        self.route = None
        self.usage = {}
        self.raw_response = None
        self.error = None
        self.confidence = 0.0

    def set_text(self, text: str):
        safe_text = text or ""
        self.response = safe_text
        self.answer = safe_text

    def get_text(self) -> str:
        return self.answer or self.response or ""
        