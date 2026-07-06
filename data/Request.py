from pydantic import BaseModel

class Request:
    def __init__(self, prompt):
        self.prompt = prompt