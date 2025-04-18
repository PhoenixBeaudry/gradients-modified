# 1. Base image with Python 3.11, CUDA 12.4, Axolotl 2.5.1
FROM --platform=linux/amd64 axolotlai/axolotl:main-latest

# 2. System libs for optimized kernels
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential cmake git libomp-dev libssl-dev && \
    rm -rf /var/lib/apt/lists/*

# 3. Install Python packages (FlashAttention, DeepSpeed, etc.)
RUN pip install --no-cache-dir \
      mlflow \
      huggingface_hub \
      wandb \
      accelerate \
      deepspeed \
      bitsandbytes \
      fairscale \
      xformers \
      optimum \
      flash-attn \
      unsloth

# 4. Prepare working directories
WORKDIR /workspace/axolotl
RUN mkdir -p \
      /workspace/axolotl/configs \
      /workspace/axolotl/outputs \
      /workspace/axolotl/data \
      /workspace/input_data

# Copy accelerate & deepspeed configs
COPY accelerate_config.yaml /workspace/axolotl/configs/accelerate_config.yaml
COPY deepspeed_stage2.json /workspace/axolotl/configs/deepspeed_stage2.json

# Set the ENV so entrypoint uses it by default
ENV ACCELERATE_CONFIG_FILE="/workspace/axolotl/configs/accelerate_config.yaml"

# 5. Default ENVs (override these in your .env)
ENV CONFIG_DIR="/workspace/axolotl/configs" \
    OUTPUT_DIR="/workspace/axolotl/outputs" \
    DATA_DIR="/workspace/axolotl/data" \
    AWS_ENDPOINT_URL="https://…r2.cloudflarestorage.com" \
    AWS_ACCESS_KEY_ID=dummy \
    AWS_DEFAULT_REGION="us-east-1" \
    AWS_SECRET_ACCESS_KEY=dummy \
    OMP_NUM_THREADS=16 \
    MKL_NUM_THREADS=16


# 6. Fake AWS creds for in‑container boto3
RUN mkdir -p /root/.aws && \
    printf "[default]\naws_access_key_id=dummy\naws_secret_access_key=dummy\n" > /root/.aws/credentials && \
    printf "[default]\nregion=us-east-1\n" > /root/.aws/config

# 7. Entrypoint: login + accelerate launch
ENTRYPOINT ["/bin/bash", "-lc", "\
  set -e; \
  echo 'Logging into HuggingFace…'; \
  if [ -n \"$HUGGINGFACE_TOKEN\" ]; then huggingface-cli login --token \"$HUGGINGFACE_TOKEN\"; fi; \
  echo 'Logging into W&B…'; \
  if [ -n \"$WANDB_TOKEN\" ]; then wandb login \"$WANDB_TOKEN\"; fi; \
  \
  # Copy dataset if provided
  if [ \"$DATASET_TYPE\" != \"hf\" ] && [ -f \"/workspace/input_data/${DATASET_FILENAME}\" ]; then \
    cp /workspace/input_data/${DATASET_FILENAME} /workspace/axolotl/data/; \
  fi; \
  \
  echo 'Starting training…'; \
  accelerate launch \
    --config_file $ACCELERATE_CONFIG_FILE \
    -m axolotl.cli.train ${CONFIG_DIR}/${JOB_ID}.yml \
"]
