FROM python:3.12-slim-trixie

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# Training accelerator backend. Defaults to Intel Arc (tested). See README
# "Training hardware": CUDA and ROCm support are best-effort/untested.
#   xpu  - Intel Arc (Level Zero + compute-runtime + torch-xpu)
#   cuda - NVIDIA (torch from PyPI; needs the NVIDIA Container Toolkit at run time)
#   rocm - AMD (torch ROCm wheels; needs a ROCm-capable host)
#   cpu  - CPU only
ARG GPU_TYPE=xpu
# ROCm wheel index version, only used when GPU_TYPE=rocm. Match this to the
# ROCm userspace version installed on the host.
ARG ROCM_VERSION=6.4

WORKDIR /app

COPY requirements.txt requirements-training.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update -qq \
    && apt-get install -y --no-install-recommends ca-certificates wget libgl1 libglib2.0-0 libxcb1 libx11-6 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Intel GPU user-space stack (Level Zero 1.34 + compute-runtime 26.35, matched to
# the torch 2.14+xpu wheels). Downloaded at build time so the source bundle stays
# small. Only installed for the Intel backend; on hosts without an Intel GPU
# torch falls back to CPU.
RUN if [ "$GPU_TYPE" = "xpu" ]; then \
        wget -q https://github.com/oneapi-src/level-zero/releases/download/v1.34.0/libze1_1.34.0%2Bu24.04_amd64.deb \
            https://github.com/intel/intel-graphics-compiler/releases/download/v2.41.5/intel-igc-core-2_2.41.5+22716_amd64.deb \
            https://github.com/intel/intel-graphics-compiler/releases/download/v2.41.5/intel-igc-opencl-2_2.41.5+22716_amd64.deb \
            https://github.com/intel/compute-runtime/releases/download/26.35.39758.10/intel-opencl-icd_26.35.39758.10-0_amd64.deb \
            https://github.com/intel/compute-runtime/releases/download/26.35.39758.10/libigdgmm12_22.10.0_amd64.deb \
            https://github.com/intel/compute-runtime/releases/download/26.35.39758.10/libze-intel-gpu1_26.35.39758.10-0_amd64.deb \
        && apt-get update -qq \
        && apt-get install -y --no-install-recommends ocl-icd-libopencl1 ./libze1_*.deb ./intel-igc-core-2_*.deb ./intel-igc-opencl-2_*.deb ./intel-opencl-icd_*.deb ./libigdgmm12_*.deb ./libze-intel-gpu1_*.deb \
        && rm -f ./*.deb && rm -rf /var/lib/apt/lists/*; \
    fi

# Training stack. The torch build depends on the target accelerator: Intel XPU
# and AMD ROCm come from PyTorch's index, CPU/CUDA from PyPI. ultralytics is
# installed afterwards so it accepts the already-present torch.
RUN if [ "$GPU_TYPE" = "xpu" ]; then \
        pip install --no-cache-dir --index-url https://download.pytorch.org/whl/xpu "torch==2.14.*" "torchvision==0.29.*"; \
    elif [ "$GPU_TYPE" = "rocm" ]; then \
        pip install --no-cache-dir --index-url https://download.pytorch.org/whl/rocm${ROCM_VERSION} torch torchvision; \
    else \
        pip install --no-cache-dir torch torchvision; \
    fi \
    && pip install --no-cache-dir -r requirements-training.txt

RUN wget -q https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt -O /app/yolo11n.pt

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
