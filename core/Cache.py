from data.Response import Response
from data.Request import Request

class Cache:
    def __init__(self):
        self.memory = {}

    def check(self, task_data):
        return None 
        
    def save(self, task_data: Request, response: Response):
        pass