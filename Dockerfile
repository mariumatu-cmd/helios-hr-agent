# syntax=docker/dockerfile:1
#
# Two stages. The builder resolves dependencies and downloads the embedding
# model; the runtime image carries only what is needed to serve. The point of
# the split is the model: fetching it at build time means a cold start on
# Render's free tier (which spins the instance down after 15 minutes idle) does
# not depend on HuggingFace being reachable, and the first user query does not
# pay an ~80 MB download.

FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# Dependency layer first: pyproject alone changes far less often than source,
# so edits to the corpus or the agent do not invalidate the pip install.
COPY pyproject.toml ./
RUN mkdir -p app agent rag/ingest mcp_server \
    && touch app/__init__.py agent/__init__.py rag/__init__.py rag/ingest/__init__.py mcp_server/__init__.py \
    && python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install . \
    && /opt/venv/bin/pip uninstall -y hr-agentic-rag
# The uninstall is deliberate. `pip install .` is the least duplicative way to
# resolve the dependency set from pyproject, but it also installs empty stub
# packages named `app`, `agent`, `rag` and `mcp_server` into site-packages.
# Those would shadow -- or worse, race with -- the real source copied into /app
# in the runtime stage. Removing the project distribution leaves exactly its
# dependencies behind, which is all this stage was for.

# Bake the ONNX weights into the image. fastembed resolves BAAI/bge-small-en-v1.5
# to its quantized variant (bge-small-en-v1.5-onnx-q), which is what the index
# was built with -- see rag/index/index_info.json.
ENV FASTEMBED_CACHE_PATH=/opt/fastembed_cache
RUN /opt/venv/bin/python -c "\
from fastembed import TextEmbedding; \
m = TextEmbedding(model_name='BAAI/bge-small-en-v1.5'); \
list(m.query_embed(['warm'])); \
print('embedding model cached')"


FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    FASTEMBED_CACHE_PATH=/opt/fastembed_cache \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    WARM_EMBEDDER=true \
    MCP_TRANSPORT=stdio

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/fastembed_cache /opt/fastembed_cache

WORKDIR /app

# The prebuilt index is committed and copied in; the service never embeds the
# corpus at boot. `build_index.py --check` in CI guarantees it matches the
# corpus, so a stale index cannot reach production.
COPY config.py ./
COPY app ./app
COPY agent ./agent
COPY rag ./rag
COPY mcp_server ./mcp_server
COPY corpus ./corpus
COPY mock_data ./mock_data
COPY evaluation ./evaluation
COPY scripts ./scripts

# Non-root: the container writes nothing outside /tmp, so there is no reason to
# run it with more privilege than it needs.
RUN useradd --create-home --uid 10001 helios && chown -R helios:helios /app
USER helios

EXPOSE 8000

# Render injects $PORT. Single worker on purpose: each worker would own its own
# MCP subprocess and its own copy of the embedding model, which does not fit in
# 512 MB. Concurrency comes from asyncio, not from processes.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
