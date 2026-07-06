from core.Cache import Cache
from core.TaskClassifier import TaskClassifier
from core.LocalRunner import LocalRunner
from core.RemoteClient import RemoteClient
from data.Request import Request
import time

class RouterEngine:
    def __init__(self, cache: Cache, classifier: TaskClassifier, local_runner: LocalRunner, remote_client: RemoteClient):
        self.cache = cache
        self.classifier = classifier
        self.local_runner = local_runner
        self.remote_client = remote_client
        self.global_timeout = 8.0

    def process_request(self, raw_task: Request):
        start_time = time.time()

        cached_response = self.cache.check(raw_task)
        if cached_response:
            return cached_response

        if self.classifier.is_local_friendly(raw_task):
            local_response = self.local_runner.generate(raw_task)
            
            if self.local_runner.is_confident(local_response):
                self.cache.save(raw_task, local_response)
                return local_response

        try:
            if time.time() - start_time > (self.global_timeout - 2.0):
                 raise TimeoutError("No time")
                 
            remote_response = self.remote_client.generate(raw_task)
            self.cache.save(raw_task, remote_response)
            return remote_response
            
        except Exception as e:
            fallback_response = self.local_runner.generate(raw_task)
            return fallback_response