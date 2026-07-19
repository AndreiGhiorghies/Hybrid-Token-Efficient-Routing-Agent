import os

from huggingface_hub import snapshot_download

# Download the embeddings classifier and the local model
# DO NOT RUN IN CONTAINER!
os.makedirs("./models/onnx", exist_ok=True)
os.makedirs("./models/gguf", exist_ok=True)

snapshot_download(
    repo_id="Xenova/all-MiniLM-L6-v2", 
    local_dir="./models/onnx",
    allow_patterns=["*.onnx", "*.json", "*.txt"]
)

snapshot_download(
    repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
    local_dir="./models/gguf",
    allow_patterns=["qwen2.5-3b-instruct-q4_k_m.gguf"]
)

