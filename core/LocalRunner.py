from data.Response import Response
from data.Request import Request

class LocalRunner:
    def __init__(self):
        pass

    def generate(self, task_data: Request) -> Response:
        return Response()

    def is_confident(self, response: Response):
        return True