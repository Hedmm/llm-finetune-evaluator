# llm-finetune-evaluator: the lfe command line in a container.
#
#   docker build -t llm-finetune-evaluator .
#   docker run --rm llm-finetune-evaluator run examples/mock_quickstart/config.yaml --report
#
# Evaluate your own config by mounting it (runs are written next to it):
#
#   docker run --rm -v "$PWD:/work" -w /work --network host \
#       llm-finetune-evaluator run config.yaml
#
# The image has no GPU stack. Build with --build-arg EXTRAS=transformers to add
# the local transformers backend (CPU-only unless you start from a CUDA image).

FROM python:3.12-slim

ARG EXTRAS=""

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LFE_CACHE_DIR=/cache

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".${EXTRAS:+[$EXTRAS]}"
COPY examples ./examples

RUN useradd --create-home --uid 1000 lfe \
    && mkdir -p /cache \
    && chown -R lfe /app /cache
USER lfe

ENTRYPOINT ["lfe"]
CMD ["--help"]
