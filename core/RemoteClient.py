from data.Response import Response
from data.Request import Request

class RemoteClient:
    def __init__(self, api_key, base_url, allowed_models):
        self.api_key = api_key
        self.base_url = base_url
        self.allowed_models = allowed_models

    def generate(self, task_data: Request):
        return Response()