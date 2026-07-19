# Hybrid Token-Efficient Routing Agent

Production-oriented submission for the Fireworks hackathon track Hybrid Token-Efficient Routing Agent.

The system is designed to maximize accuracy while minimizing remote token usage by combining an embeddings-based task router, local inference, and controlled Fireworks fallback.

## Objective

The container processes a fixed task set from /input/tasks.json and writes final answers to /output/results.json.

Scoring depends on:

- passing the accuracy gate
- then minimizing tokens consumed through FIREWORKS_BASE_URL

Local inference contributes to accuracy and consumes zero Fireworks tokens.

## End-to-End Flow

1. On startup, tasks are loaded from /input/tasks.json.
2. Each prompt is classified by an embeddings classifier into category plus difficulty.
3. A routing decision is made per task: local-friendly tasks are queued for local inference, the rest are queued for remote inference.
4. Remote tasks are sent first, in batches, to reduce request overhead and prompt duplication.
5. Any remote failures are moved to local fallback.
6. Local tasks are executed with category-specific prompts.
7. Any local answers below the confidence threshold are retried once through remote fallback.
8. Results are normalized and written to /output/results.json.

Global execution is protected by a hard time budget (9.5 minutes), aligned with the challenge runtime constraints.

## Task Classification and Routing

Routing is driven by an ONNX embeddings classifier that compares each incoming prompt against anchored examples.

For every task, the router derives:

- capability category
- estimated difficulty
- local-friendliness decision

This allows the system to reserve remote tokens for tasks where local completion is less reliable, while still preserving a high hit rate on easier prompts.

## Remote Inference Strategy (Token Efficiency)

Remote requests are grouped in batches to reduce total request count and repeated instruction overhead.

For Math and Logical tasks, the prompts enforce a response style: return Python code that computes the final answer. The generated code is executed, and the execution output is used as the answer when successful. This reduces the need for long natural-language reasoning traces and improves token efficiency.

Before batching, tasks are split into two remote queues:

- code queue: Math, Logical, Code Generation, Code Debug
- text queue: Factual, Sentiment, Summarisation, NER

The two queues are routed with different model priorities:

- code queue priority: Kimi family first, then Minimax, then Gemma-family options when present in ALLOWED_MODELS
- text queue priority: Minimax family first, then Kimi, then Gemma-family options when present in ALLOWED_MODELS

This policy is intentional: Kimi models generally perform better on code-heavy tasks, while Minimax is prioritized for general text tasks.

For each call, the runtime validates model eligibility by selecting the first candidate that is also present in ALLOWED_MODELS.

For multi-task batches, the response contract is strict:

@@[task_id]|[answer]

That compact delimiter-based format enables deterministic parsing and avoids unnecessary output verbosity.

The implementation also supports cumulative remote token accounting (prompt/completion/total) through the usage-tracking option in the remote client, useful for local optimization runs.

## Local Inference Strategy

Local inference uses the bundled GGUF model(qwen2.5-3b-instruct-q4_k_m) and category-specific system prompts.

For Math and Logical tasks, local prompts enforce the same response style: Python code that calculates and prints the final answer.

Local answer acceptance is confidence-gated using model log-probability-derived confidence. Low-confidence local outputs are escalated once to remote fallback.

## Runtime Constraints

The solution is tuned for the hackathon evaluation envelope:

- 2 vCPU
- 4 GB RAM

Container resource limits can be reproduced locally during validation using Docker runtime flags.

## Build and Run

Important: run init.py before building the image.

init.py downloads required local assets (including qwen2.5-3b-instruct-q4_k_m.gguf) into models so they are copied into the image at build time. Models are not downloaded at container runtime.

### Step 1: Download model assets

~~~bash
python init.py
~~~

### Step 2: Build image

~~~bash
docker buildx build --platform linux/amd64 -t [your_username]/router:v1 --push .
~~~

### Step 3: Run with hackathon-like limits and env vars

~~~bash
docker run --rm -it --memory="4g" --memory-swap="4g" --cpus="2" -v "%cd%/input:/input" -v "%cd%/output:/output" -e FIREWORKS_API_KEY="<your_fireworks_key>" -e FIREWORKS_BASE_URL="https://api.fireworks.ai/inference/v1" -e ALLOWED_MODELS="kimi-k2p7-code,minimax-m3" [your_username]/router:v1
~~~

## Input and Output Contract

Input file: /input/tasks.json

~~~json
[
  { "task_id": "t1", "prompt": "..." },
  { "task_id": "t2", "prompt": "..." }
]
~~~

Output file: /output/results.json

~~~json
[
  { "task_id": "t1", "answer": "..." },
  { "task_id": "t2", "answer": "..." }
]
~~~

Output is written once processing is complete, and the program exits with code 0 on successful completion.




