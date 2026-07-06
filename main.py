from core.RouterEngine import RouterEngine
from core.TaskClassifier import TaskClassifier
from core.LocalRunner import LocalRunner
from core.RemoteClient import RemoteClient

import json
import time
import os
import sys
import concurrent.futures

def main():
    GLOBAL_START_TIME = time.time()
    
    GLOBAL_TIME_LIMIT = 9.5 * 60 

    api_key = os.environ.get("FIREWORKS_API_KEY")
    base_url = os.environ.get("FIREWORKS_BASE_URL")
    allowed_models = os.environ.get("ALLOWED_MODELS", "").split(",")

    engine = RouterEngine(
        classifier=TaskClassifier(),
        local_runner=LocalRunner(),
        remote_client=RemoteClient(api_key, base_url, allowed_models)
    )

    with open("/input/tasks.json", "r") as f:
        tasks = json.load(f)

    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        for item in tasks:
            task_id = item["task_id"]
            prompt = item["prompt"]
            
            if time.time() - GLOBAL_START_TIME > GLOBAL_TIME_LIMIT:
                results.append({
                    "task_id": task_id,
                    "answer": "Error: Global timeout limit reached."
                })
                continue

            future = executor.submit(engine.process_request, prompt)
            
            try:
                answer = future.result(timeout=25)
            except concurrent.futures.TimeoutError:
                answer = "Error: Per-task timeout."
            except Exception as e:
                answer = "Error: Internal exception."

            results.append({
                "task_id": task_id,
                "answer": answer
            })

    os.makedirs("/output", exist_ok=True)
    with open("/output/results.json", "w") as f:
        json.dump(results, f)

    sys.exit(0)

if __name__ == "__main__":
    main()