# Thin application image. Consumes the published base runtime (stock Python +
# GPU user-space + torch/ultralytics) and only adds the app code and its small
# Python deps, so per-commit builds stay fast. Rebuild/publish the base via
# Dockerfile.base and bump BASE_VERSION when the base stack changes.
ARG BASE_IMAGE=ghcr.io/vu-ops/frigate-errata-base-xpu:1

FROM ${BASE_IMAGE}

# App version, injected by CI (the image tag, e.g. 0.1.10).
ARG VERSION=0.0.0
ENV ERRATA_VERSION=${VERSION}

LABEL org.opencontainers.image.title="frigate-errata" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.source="https://github.com/vu-ops/frigate-errata" \
      org.opencontainers.image.licenses="MIT"

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY errata/ ./errata/

RUN useradd --system --uid 1000 errata \
    && mkdir -p /data /publish \
    && chown -R errata:errata /data /publish

USER errata

EXPOSE 8501
VOLUME ["/data", "/publish"]

HEALTHCHECK --interval=60s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8501/healthz', timeout=4)" || exit 1

CMD ["python", "-m", "errata"]
