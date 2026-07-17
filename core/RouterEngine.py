from core.TaskClassifier import TaskClassifier
from core.LocalRunner import LocalRunner
from core.RemoteClient import RemoteClient
from data.Request import Request
from data.Response import Response


class RouterEngine:
    def __init__(
        self,
        classifier: TaskClassifier,
        local_runner: LocalRunner,
        remote_client: RemoteClient
    ):
        self.classifier = classifier
        self.local_runner = local_runner
        self.remote_client = remote_client
        self.allowed_models = remote_client.allowed_models
        self.tasks_to_api = dict()

    def get_answers(self) -> list[Response]:
        fallback_tasks_list = []
        answers = []

        remote_answers = self.remote_client.process_batches(fallback_tasks_list)
        answers.extend(remote_answers)

        for task in fallback_tasks_list:
            self.local_runner.add_task(task)

        fallback_tasks_list.clear()
        local_answers = self.local_runner.process_tasks(fallback_tasks_list)
        answers.extend(local_answers)

        for task in fallback_tasks_list:
            self.remote_client.add_task(task)

        fallback_tasks_list.clear()
        remote_answers_final = self.remote_client.process_batches(fallback_tasks_list)
        answers.extend(remote_answers_final)

        for task in fallback_tasks_list:
            answers.append(Response(id = task.id, success=False, answer = "Unable to determine a confident answer."))

        return answers

    def process_request(self, task: Request) -> None:
        self.classifier.classify(task)

        if self.classifier.is_local_friendly(task):
            self.local_runner.add_task(task)
        else:
            self.remote_client.add_task(task)