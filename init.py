from huggingface_hub import snapshot_download

# Download the embeddings classifier
# DO NOT RUN IN CONTAINER!
snapshot_download(
    repo_id="Xenova/all-MiniLM-L6-v2", 
    local_dir="./models/onnx",
    allow_patterns=["*.onnx", "*.json", "*.txt"]
)