# 1. Base image with Python 3.11, CUDA 12.4, Axolotl 2.5.1
FROM --platform=linux/amd64 axolotlai/axolotl:main-latest

# 2. System libs for optimized kernels
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential cmake git libomp-dev libssl-dev && \
    rm -rf /var/lib/apt/lists/*

# 3. Install Python packages (FlashAttention, DeepSpeed, etc.)
RUN pip install --no-cache-dir --no-build-isolation \
      mlflow \
      huggingface_hub \
      wandb \
      deepspeed \
      protobuf \
      liger-kernel \
      triton

# 4. Prepare working directories
WORKDIR /workspace/axolotl
RUN mkdir -p \
      /workspace/axolotl/configs \
      /workspace/axolotl/outputs \
      /workspace/axolotl/data \
      /workspace/input_data

# 5. Default ENVs (override these in your .env)
ENV CONFIG_DIR="/workspace/axolotl/configs" \
    OUTPUT_DIR="/workspace/axolotl/outputs" \
    DATA_DIR="/workspace/axolotl/data" \
    AWS_ENDPOINT_URL="https://5a301a635a9d0ac3cb7fcc3bf373c3c3.r2.cloudflarestorage.com" \
    AWS_ACCESS_KEY_ID=d49fdd0cc9750a097b58ba35b2d9fbed \
    AWS_DEFAULT_REGION="us-east-1" \
    AWS_SECRET_ACCESS_KEY=02e398474b783af6ded4c4638b5388ceb8079c83bb2f8233d5bcef0e60addba6 \
    OMP_NUM_THREADS=16 \
    MKL_NUM_THREADS=16


# 6. Fake AWS creds for in‑container boto3
RUN mkdir -p /root/.aws && \
    printf "[default]\naws_access_key_id=dummy\naws_secret_access_key=dummy\n" > /root/.aws/credentials && \
    printf "[default]\nregion=us-east-1\n" > /root/.aws/config

# from the root of your repo, so setup.py / pyproject.toml is present
WORKDIR /workspace/axolotl
# install your local code as a package
RUN pip install --no-cache-dir .

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
  axolotl train ${CONFIG_DIR}/${JOB_ID}.yml \
"]
