from core.RouterEngine import RouterEngine
from core.Cache import Cache
from core.TaskClassifier import TaskClassifier
from core.LocalRunner import LocalRunner
from core.RemoteClient import RemoteClient
from data.Request import Request

class Evaluator:
    def __init__(self):
        self.engine = RouterEngine(
            cache=Cache(),
            classifier=TaskClassifier(),
            local_runner=LocalRunner(),
            remote_client=RemoteClient()
        )

    def run():
        pass