from core.RouterEngine import RouterEngine
from core.Cache import Cache
from core.TaskClassifier import TaskClassifier
from core.LocalRunner import LocalRunner
from core.RemoteClient import RemoteClient
from data.Request import Request

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

engine = RouterEngine(
    cache=Cache(),
    classifier=TaskClassifier(),
    local_runner=LocalRunner(),
    remote_client=RemoteClient()
)

class TaskRequest(BaseModel):
    task: str

@app.post("/run")
def process_task(request: TaskRequest):
    answer = engine.process_request(Request(request.task))
    
    return {"answer": answer}

""" if __name__ == "__main__":
    engine = RouterEngine(
        cache=Cache(),
        classifier=TaskClassifier(),
        local_runner=LocalRunner(),
        remote_client=RemoteClient()
    )

    print(">> RESPONSE:", engine.process_request(Request("How much is 2+2?")))
    
    print(">> RESPONSE:", engine.process_request(Request("Write an AST parsing algorithm in Python and create a comparative performance chart across several scenarios."))) """