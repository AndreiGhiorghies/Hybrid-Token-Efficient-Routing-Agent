from data.Request import Request

class TaskClassifier:
    def is_local_friendly(self, task_data: Request):
        word_count = len(task_data.prompt.split())
        return word_count < 50