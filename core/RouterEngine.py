from core.TaskClassifier import TaskClassifier
from core.LocalRunner import LocalRunner
from core.RemoteClient import RemoteClient
from data.Request import Request

class RouterEngine:
    def __init__(self, classifier: TaskClassifier, local_runner: LocalRunner, remote_client: RemoteClient):
        self.classifier = classifier
        self.local_runner = local_runner
        self.remote_client = remote_client

    def process_request(self, raw_task: Request):
        if self.classifier.is_local_friendly(raw_task):
            local_response = self.local_runner.generate(raw_task)
            
            if local_response.success and self.local_runner.is_confident(local_response):
                return local_response

        remote_response = self.remote_client.generate(raw_task)
        if remote_response.success:
            return remote_response
        
        fallback_response = self.local_runner.generate(raw_task)
        return fallback_response