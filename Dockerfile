FROM python:3.10-slim AS builder

ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc git python3-dev rsync && \
    rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip

RUN pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

RUN git clone https://github.com/run-llama/llama_index.git /app/llama_index
COPY llama_index_extra/ /tmp/llama_index_extra/
RUN rsync -a /tmp/llama_index_extra/ /app/llama_index/

RUN pip install -e /app/llama_index/llama-index-core
RUN pip install -e /app/llama_index/llama-index-integrations/embeddings/llama-index-embeddings-cohere
RUN pip install -e /app/llama_index/llama-index-integrations/embeddings/llama-index-embeddings-huggingface
RUN pip install -e /app/llama_index/llama-index-integrations/indices/llama-index-indices-managed-bge-m3
RUN pip install -e /app/llama_index/llama-index-integrations/indices/llama-index-indices-managed-colbert
RUN pip install -e /app/llama_index/llama-index-integrations/retrievers/llama-index-retrievers-bm25

COPY . /app/LRG

RUN mv /app/LRG/test_data /app && \
    python /app/LRG/setup_data.py

FROM python:3.10-slim AS runtime

WORKDIR /app
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app/llama_index /app/llama_index
COPY --from=builder /app/LRG /app/LRG
COPY --from=builder /app/test_data /app/test_data

CMD ["bash"]