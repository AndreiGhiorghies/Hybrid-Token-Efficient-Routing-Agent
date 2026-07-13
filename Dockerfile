FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    libgomp1 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# instalare wheel precompilat, CPU generic — nu mai are nevoie de build-essential/cmake
RUN pip install --no-cache-dir llama-cpp-python==0.3.33 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]