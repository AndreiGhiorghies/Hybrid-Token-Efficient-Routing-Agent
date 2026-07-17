from core.RouterEngine import RouterEngine
from core.TaskClassifier import TaskClassifier
from core.LocalRunner import LocalRunner
from core.RemoteClient import RemoteClient
from data.Request import Request

import json
import time
import os
import sys


INPUT_PATH = "/input/tasks.json"
OUTPUT_PATH = "/output/results.json"

GLOBAL_TIME_LIMIT_SECONDS = 9.5 * 60

def main():
    global_start_time = time.time()

    """ api_key = os.environ.get("FIREWORKS_API_KEY", "")
    base_url = os.environ.get("FIREWORKS_BASE_URL", "") 
    allowed_models = [
        model.strip()
        for model in os.environ.get("ALLOWED_MODELS", "").split(",")
    ] """

    api_key = "fw_QkCH6xgu4f7jSusbRmsft4"
    base_url = "https://api.fireworks.ai/inference/v1/"
    allowed_models = ["minimax-m3", "kimi-k2p7-code", "accounts/fireworks/models/gemma-4-31b-it", "accounts/fireworks/models/gemma-4-31b-it-nvfp4", "accounts/fireworks/models/gemma-4-26b-a4b-it"]

    engine = RouterEngine(
        classifier=TaskClassifier(),
        local_runner=LocalRunner(global_start_time, GLOBAL_TIME_LIMIT_SECONDS),
        remote_client=RemoteClient(api_key, base_url, allowed_models, global_start_time, GLOBAL_TIME_LIMIT_SECONDS)
    )

    try:
        with open(INPUT_PATH, "r", encoding="utf-8") as f:
            tasks = json.load(f)
    except Exception:
        os.makedirs("/output", exist_ok=True)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump([], f)
        
        sys.exit(0)

    for item in tasks:
        task_id = item.get("task_id", "")
        prompt = item.get("prompt", "")

        request = Request(task_id, prompt)
        engine.process_request(request)

    results = engine.get_answers()

    results_dicts = [
        {
            "task_id": response.id,
            "answer": response.answer,
        }
        for response in results
    ]

    os.makedirs("/output", exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results_dicts, f, ensure_ascii=False)

    sys.exit(0)


if __name__ == "__main__":
    main()

# docker buildx build --platform linux/amd64 -t andrei010/hackathon-router:latest --push .

# docker run --rm -it --memory="4g" --memory-swap="4g" --cpus="2" -v "%cd%/temp/mock_harness/input:/input" -v "%cd%/temp/mock_harness/output:/output" -e FIREWORKS_API_KEY="fw_QkCH6xgu4f7jSusbRmsft4" -e FIREWORKS_BASE_URL="https://api.fireworks.ai/inference/v1" -e ALLOWED_MODELS="kimi-k2p7-code" --entrypoint /bin/bash andrei010/hackathon-router:v10

# tot la kimi cu batch: 711 + 849 = 1560 tokens
# kimi + minimax: 501 + 1277 = 1778 tokens