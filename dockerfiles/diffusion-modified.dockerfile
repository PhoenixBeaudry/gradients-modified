# Use your Kohya base image (Python, PyTorch, sd-scripts, etc.)
FROM --platform=linux/amd64 diagonalge/kohya_latest:latest

USER root

# 1. Install system tools needed for the CUDA runfile and builds
RUN apt-get update && apt-get install -y --no-install-recommends \
      wget ca-certificates build-essential libomp-dev cmake git \
    && rm -rf /var/lib/apt/lists/*

# 2. Install the CUDA Toolkit (including nvcc) via NVIDIA runfile
ENV CUDA_VERSION=12.8.1
ENV CUDA_RUNFILE=cuda_${CUDA_VERSION}.0_570.124.06_linux.run
RUN wget -O /tmp/$CUDA_RUNFILE https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux.run && \
    sh /tmp/$CUDA_RUNFILE --silent --toolkit

# 3. Set CUDA environment
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# 4. Prepare training directories
RUN mkdir -p /dataset/configs /dataset/outputs /dataset/images /dataset/latents && \
    chmod -R 777 /dataset

# 5. Copy in your Accelerate & DeepSpeed config files
COPY accelerate_config.yaml  /workspace/accelerate_config.yaml
COPY deepspeed_diffusion.json /workspace/deepspeed_diffusion.json

# 6. Install ML Python packages
RUN pip install --no-cache-dir \
      xformers \
      flash_attn \
      diffusers[torch] \
      accelerate \
      deepspeed \
      transformers \
      peft

# 7. Set up environment variables for your training entrypoint
ENV CONFIG_DIR="/dataset/configs"
ENV OUTPUT_DIR="/dataset/outputs"
ENV DATASET_DIR="/dataset/images"
ENV LATENT_DIR="/dataset/latents"
ENV ACCELERATE_CONFIG_FILE="/workspace/accelerate_config.yaml"
ENV DEEPSPEED_CONFIG_FILE="/workspace/deepspeed_diffusion.json"
ENV OMP_NUM_THREADS=16
ENV MKL_NUM_THREADS=16

# 8. Entrypoint to log in once and launch across 8 A100s
ENTRYPOINT ["/bin/bash","-lc","\
  set -e; \
  echo 'Logging into HuggingFace…'; \
  huggingface-cli login --token \"$HUGGINGFACE_TOKEN\"; \
  echo 'Logging into W&B…'; \
  wandb login \"$WANDB_TOKEN\"; \
  echo 'Starting diffusion training…'; \
  accelerate launch \
    --config_file $ACCELERATE_CONFIG_FILE \
    --dynamo_backend no \
    --dynamo_mode default \
    --mixed_precision bf16 \
    --num_processes 8 \
    --num_machines 1 \
    --num_cpu_threads_per_process 4 \
    -m /app/sd-scripts/${BASE_MODEL}_train_network.py \
    --config_file ${CONFIG_DIR}/${JOB_ID}.toml \
"]
