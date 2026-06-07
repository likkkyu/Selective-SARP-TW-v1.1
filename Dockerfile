FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-venv \
    ca-certificates \
    git \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/Project-VRP_v6_4

COPY environment.yml ./

RUN python -m pip install --upgrade pip && \
    python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
        torch \
        torchvision && \
    python -m pip install \
        'numpy>=1.23' \
        'scipy>=1.10' \
        'tqdm>=4.65' \
        tensorboard_logger \
        tensorboard \
        'gurobipy>=10.0' \
        'matplotlib>=3.7' \
        'seaborn>=0.12' \
        'scikit-learn>=1.3'

COPY . .

ENV PYTHONPATH=/workspace/Project-VRP_v6_4

CMD ["python", "run_training_optimized.py", "--graph_sizes", "25", "50", "100", "--calibrate-before-train"]
