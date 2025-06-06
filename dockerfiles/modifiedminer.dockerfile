# 1. Base image with Python 3.11, CUDA 12.4, Axolotl 2.5.1
FROM axolotlai/axolotl:main-20250420-py3.11-cu124-2.5.1

# 3. Install Python packages
RUN pip install --no-build-isolation mlflow huggingface_hub wandb

# 4. Prepare working directories
WORKDIR /workspace/axolotl
RUN mkdir -p /workspace/axolotl/configs \
    /workspace/axolotl/outputs \
    /workspace/axolotl/data \
    /workspace/input_data

# 5. Default ENVs (override these in your .env)
ENV CONFIG_DIR="/workspace/axolotl/configs"
ENV OUTPUT_DIR="/workspace/axolotl/outputs"
ENV AWS_ENDPOINT_URL="https://5a301a635a9d0ac3cb7fcc3bf373c3c3.r2.cloudflarestorage.com"
ENV AWS_ACCESS_KEY_ID=d49fdd0cc9750a097b58ba35b2d9fbed
ENV AWS_DEFAULT_REGION="us-east-1"
ENV AWS_SECRET_ACCESS_KEY=02e398474b783af6ded4c4638b5388ceb8079c83bb2f8233d5bcef0e60addba6
ENV OMP_NUM_THREADS=16 
ENV MKL_NUM_THREADS=16


# 6. Fake AWS creds for in‑container boto3
RUN mkdir -p /root/.aws && \
    echo "[default]\naws_access_key_id=dummy_access_key\naws_secret_access_key=dummy_secret_key" > /root/.aws/credentials && \
    echo "[default]\nregion=us-east-1" > /root/.aws/config

    CMD echo 'Preparing data...' && \
    pip install --no-build-isolation mlflow && \
    pip install --no-build-isolation protobuf && \
    pip install --no-build-isolation --upgrade huggingface_hub && \
    if [ -n "$HUGGINGFACE_TOKEN" ]; then \
    echo "Attempting to log in to Hugging Face" && \
    huggingface-cli login --token "$HUGGINGFACE_TOKEN" --add-to-git-credential; \
    else \
    echo "HUGGINGFACE_TOKEN is not set. Skipping Hugging Face login."; \
    fi && \
    if [ -n "$WANDB_TOKEN" ]; then \
    echo "Attempting to log in to W&B" && \
    wandb login "$WANDB_TOKEN"; \
    else \
    echo "WANDB_TOKEN is not set. Skipping W&B login."; \
    fi && \
    if [ "$DATASET_TYPE" != "hf" ] && [ -f "/workspace/input_data/${DATASET_FILENAME}" ]; then \
    cp /workspace/input_data/${DATASET_FILENAME} /workspace/axolotl/data/${DATASET_FILENAME}; \
    cp /workspace/input_data/${DATASET_FILENAME} /workspace/axolotl/${DATASET_FILENAME}; \
    fi && \
    echo 'Starting training command' && \
    axolotl train ${CONFIG_DIR}/${JOB_ID}.yml
