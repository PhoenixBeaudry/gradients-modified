# — 1) Base image
FROM --platform=linux/amd64 diagonalge/kohya_latest:latest

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

# — 10) Write an entrypoint that exports and then execs accelerate
RUN printf '%s\n' \
  '#!/usr/bin/env bash' \
  'set -e' \
  '' \
  '# (re‑export to ensure all children see them)' \
  'export NCCL_IB_DISABLE=$NCCL_IB_DISABLE' \
  'export NCCL_DEBUG=$NCCL_DEBUG' \
  'export NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME' \
  'export OMP_NUM_THREADS=$OMP_NUM_THREADS' \
  'export MKL_NUM_THREADS=$MKL_NUM_THREADS' \
  'export HF_DATASETS_USE_PREFETCH=$HF_DATASETS_USE_PREFETCH' \
  '' \
  'exec accelerate launch \' \
  '  --num_processes 1 \' \
  '  --num_machines 1 \' \
  '  --mixed_precision bf16 \' \
  '  /app/sd-scripts/${BASE_MODEL}_train_network.py \' \
  '  --config_file ${CONFIG_DIR}/${JOB_ID}.toml' \
  > /entrypoint.sh && chmod +x /entrypoint.sh

# — 11) Use our script as PID 1
ENTRYPOINT ["/entrypoint.sh"]
