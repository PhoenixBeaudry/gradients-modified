# — 1) Base image
FROM diagonalge/kohya_latest:latest

# — 2) Switch to root to install system deps
USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      git \
      build-essential \
      libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# — 3) Create & activate a dedicated venv
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# — 4) Upgrade pip/build‑tools & install Cython
RUN pip install --no-cache-dir --upgrade pip setuptools wheel cython


# — 6) Install optimized backends
RUN pip install --no-cache-dir \
      xformers \
      bitsandbytes


# — 8) Prepare dataset directories
RUN mkdir -p /dataset/{configs,outputs,images} \
 && chmod -R 777 /dataset

# — 9) Permanent ENV tweaks for performance
ENV CONFIG_DIR="/dataset/configs" \
    OUTPUT_DIR="/dataset/outputs" \
    DATASET_DIR="/dataset/images" \
    NCCL_IB_DISABLE=0 \
    NCCL_DEBUG=INFO \
    NCCL_SOCKET_IFNAME=^docker0,lo \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    HF_DATASETS_USE_PREFETCH=1


CMD accelerate launch --mixed_precision bf16 --multi_gpu --num_processes 8 --num_machines 1 --num_cpu_threads_per_process 4 /app/sd-scripts/${BASE_MODEL}_train_network.py --config_file ${CONFIG_DIR}/${JOB_ID}.toml