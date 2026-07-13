from core.RouterEngine import RouterEngine
from core.TaskClassifier import TaskClassifier
from core.LocalRunner import LocalRunner
from core.RemoteClient import RemoteClient
from data.Request import Request
from data.Response import Response

import json
import time
import os
import sys
import traceback


INPUT_PATH = "/input/tasks.json"
OUTPUT_PATH = "/output/results.json"

GLOBAL_TIME_LIMIT_SECONDS = 9.5 * 60


def extract_answer(result) -> str:
    if result is None:
        return "Unable to determine a confident answer."

    if isinstance(result, Response):
        text = result.get_text()
        return text if text else "Unable to determine a confident answer."

    if hasattr(result, "get_text"):
        text = result.get_text()
        return text if text else "Unable to determine a confident answer."

    if hasattr(result, "answer"):
        text = getattr(result, "answer")
        return str(text) if text else "Unable to determine a confident answer."

    if hasattr(result, "response"):
        text = getattr(result, "response")
        return str(text) if text else "Unable to determine a confident answer."

    return str(result)


def main():
    global_start_time = time.time()

    save_tasks = []

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
        local_runner=LocalRunner(),
        remote_client=RemoteClient(api_key, base_url, allowed_models)
    )

    try:
        with open(INPUT_PATH, "r", encoding="utf-8") as f:
            tasks = json.load(f)
    except Exception:
        os.makedirs("/output", exist_ok=True)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump([], f)
        sys.exit(0)

    results = []

    for item in tasks:
        task_id = item.get("task_id", "")
        prompt = item.get("prompt", "")

        save_tasks.append({
            "task_id": task_id,
            "prompt": prompt
        })

        try:
            request = Request(task_id, prompt)
            engine.process_request(request)
        except Exception:
            results.append({
                "task_id": task_id,
                "answer": "Unable to determine a confident answer."
            })

    engine.get_answers(results)

    for i in range(len(engine.local_runner.tasks_to_local)):
        elapsed = time.time() - global_start_time
        remaining = GLOBAL_TIME_LIMIT_SECONDS - elapsed


        task_data = engine.local_runner.tasks_to_local[i]
        if remaining < 30:
            results.append({
                "task_id": task_data.id,
                "answer": "Unable to determine a confident answer within the time limit."
            })
            continue
        response = engine.local_runner.get_answer(i)
        if response is not None and response.success:
            answer = extract_answer(response)
            results.append({
                "task_id": task_data.id,
                "answer": answer
            })
        else:
            results.append({
                "task_id": task_data.id,
                "answer": "Unable to determine a confident answer."
            })

    """ for item in tasks:
        task_id = item.get("task_id", "")
        prompt = item.get("prompt", "")

        save_tasks.append({
            "task_id": task_id,
            "prompt": prompt
        })

        elapsed = time.time() - global_start_time
        remaining = GLOBAL_TIME_LIMIT_SECONDS - elapsed

        if remaining < 30:
            results.append({
                "task_id": task_id,
                "answer": "Unable to determine a confident answer within the time limit."
            })
            continue

        try:
            request = Request(task_id, prompt)
            response = engine.process_request(request)
            if response is not None and response.success:
                #print(f"Processed task_id={task_id} with model {response.model if response else 'None'} and route {response.route if response else 'None'} and usage {response.usage if response else 'None'} with response: {response.get_text() if response else 'None'}")
                answer = extract_answer(response)
                results.append({
                    "task_id": task_id,
                    "answer": answer
                })

        except Exception:
            traceback.print_exc(file=sys.stderr)
     """        ##print(f"Error processing task_id={task_id}. Adding fallback answer.")
    """ answer = "Unable to determine a confident answer. la exceptie"
    results.append({
        "task_id": task_id,
        "answer": answer
    }) """

    #print("\n\nINAINTE: ", results, "\n\n")
    #print("\n\nDUPA: ", results)

    for task in save_tasks:
        if not any(r["task_id"] == task["task_id"] for r in results):
            results.append({
                "task_id": task["task_id"],
                "answer": "Unable to determine a confident answer."
            })

    os.makedirs("/output", exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)

    sys.exit(0)


if __name__ == "__main__":
    main()

# docker buildx build --platform linux/amd64 -t andrei010/hackathon-router:latest --push .

# docker run --rm -it --memory="4g" --memory-swap="4g" --cpus="2" -v "%cd%/temp/mock_harness/input:/input" -v "%cd%/temp/mock_harness/output:/output" -e FIREWORKS_API_KEY="fw_QkCH6xgu4f7jSusbRmsft4" -e FIREWORKS_BASE_URL="https://api.fireworks.ai/inference/v1" -e ALLOWED_MODELS="kimi-k2p7-code" --entrypoint /bin/bash andrei010/hackathon-router:v10

# tot la kimi cu batch: 711 + 849 = 1560 tokens