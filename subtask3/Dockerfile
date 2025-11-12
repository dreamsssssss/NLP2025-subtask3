FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /workspace

ENV HF_HOME=/root/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COPY requirements.txt .
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY app.py .

CMD ["/bin/bash"]
