# CUDA 12.8 for Blackwell (sm_120). The base's torch 2.7.1+cu128 satisfies requirements.txt's
# torch pins, so uv keeps it instead of pulling PyPI's cu126 build.
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Europe/Amsterdam \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUTLASS_PATH=/opt/cutlass \
    CUDA_HOME=/usr/local/cuda \
    PROTENIX_ROOT_DIR=/cache \
    PATH=/usr/local/cuda/bin:${PATH}

ARG BUILD_CCD_CACHE=0

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        g++ \
        gcc \
        libc6-dev \
        make \
        ninja-build \
        postgresql \
        hmmer \
        kalign \
        libsm6 \
        libxext6 \
        libxrender1 \
        wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Set working directory
WORKDIR /app

# Install Python dependencies
# Copy requirements.txt first to leverage Docker cache
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Clone CUTLASS
RUN git clone -b v3.5.1 https://github.com/NVIDIA/cutlass.git /opt/cutlass

# Copy the rest of the application code and install the package
COPY . .
RUN uv pip install --system --no-cache .

# Prebuild Protenix's custom CUDA layer-norm extension into the installed
# site-packages location so ephemeral `docker compose run --rm` containers do
# not JIT-compile it on first import.
RUN cd /tmp && \
    python -c "import protenix; print(protenix.__file__); from protenix.model.layer_norm.layer_norm import fast_layer_norm_cuda_v2; print(fast_layer_norm_cuda_v2.__file__)"

# Optionally precompute the CCD cache during image build. The upstream docs treat
# this as data-root preparation rather than a required install step, so default to
# skipping it and allow opting in with --build-arg BUILD_CCD_CACHE=1.
RUN mkdir -p ${PROTENIX_ROOT_DIR}/common && \
    if [ "${BUILD_CCD_CACHE}" = "1" ]; then \
        python scripts/gen_ccd_cache.py -c ${PROTENIX_ROOT_DIR}/common -n 1; \
    fi

# Verify the package installed without triggering first-run CUDA kernel builds.
RUN python -c "import importlib.metadata as metadata; print(metadata.version('protenix'))"
