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

    api_key = os.environ.get("FIREWORKS_API_KEY", "")
    base_url = os.environ.get("FIREWORKS_BASE_URL", "")
    allowed_models = [
        model.strip()
        for model in os.environ.get("ALLOWED_MODELS", "").split(",")
        if model.strip()
    ]

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
            answer = extract_answer(response)

        except Exception:
            traceback.print_exc(file=sys.stderr)
            answer = "Unable to determine a confident answer."

        results.append({
            "task_id": task_id,
            "answer": answer
        })

    os.makedirs("/output", exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)

    sys.exit(0)


if __name__ == "__main__":
    main()