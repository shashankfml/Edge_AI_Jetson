# Stage 1: Base system setup with JetPack version selection
ARG JETPACK_VERSION=r35.1.0
FROM nvcr.io/nvidia/l4t-base:${JETPACK_VERSION} as base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TZ=Etc/UTC \
    PATH="/root/.local/bin:$PATH" \
    PYTHONPATH=/usr/lib/python3.8:/usr/local/lib/python3.8/dist-packages

# Python 3.8 and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.8 \
    python3.8-dev \
    python3.8-distutils \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libjpeg-dev \
    libpng-dev \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 \
    && ln -s /usr/bin/python3.8 /usr/local/bin/python \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Builder
FROM base AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

ARG JETPACK_VERSION

# Install PyTorch + torchvision matching JetPack version
RUN if [ "$JETPACK_VERSION" = "r32.7.1" ]; then \
        pip3 install --user torch==1.9.0 torchvision==0.10.0 --extra-index-url https://download.pytorch.org/whl/cu111; \
    elif [ "$JETPACK_VERSION" = "r35.3.1" ]; then \
        pip3 install --user torch==2.0.0 torchvision==0.15.1 --extra-index-url https://download.pytorch.org/whl/cu118; \
    else \
        pip3 install --user torch torchvision; \
    fi

# Install other dependencies
COPY requirements.txt .
RUN pip3 install --user --no-cache-dir --upgrade pip && \
    pip3 install --user --no-cache-dir -r requirements.txt

# Stage 3: Runtime
FROM base AS runtime

# Copy built packages and app files
COPY --from=builder /root/.local /root/.local
COPY . /app
COPY node_exporter /usr/local/bin/node_exporter
COPY exporter.py .
COPY final_app.py .

# Prepare directories
RUN mkdir -p /app/process_video \
    /app/feedback/images \
    /app/feedback/labels \
    && chmod +x /usr/local/bin/node_exporter

WORKDIR /app

EXPOSE 8501 8200 9200

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["sh", "-c", "/usr/local/bin/node_exporter --web.listen-address=0.0.0.0:9200 & \
           python3 exporter.py & \
           streamlit run final_app.py --server.port=8501 --server.address=0.0.0.0"]

# # Stage 1: Base system setup with JetPack version selection
# ARG JETPACK_VERSION=r32.7.1
# FROM nvcr.io/nvidia/l4t-base:${JETPACK_VERSION} as base

# ENV DEBIAN_FRONTEND=noninteractive \
#     PYTHONUNBUFFERED=1 \
#     TZ=Etc/UTC \
#     PATH="/root/.local/bin:$PATH" \
#     # Force Python 3.8
#     PYTHON_VERSION=3.8 \
#     PYTHONPATH=/usr/lib/python3.8:/usr/local/lib/python3.8/dist-packages

# # Install specific Python 3.8 and dependencies
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     python3.8 \
#     python3.8-dev \
#     python3.8-distutils \
#     python3-pip \
#     && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 \
#     && ln -s /usr/bin/python3.8 /usr/local/bin/python \
#     && apt-get purge -y python3.9* python3.10* \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

# # Common dependencies across JetPack versions
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     libgl1-mesa-glx \
#     libglib2.0-0 \
#     libsm6 \
#     libxext6 \
#     libxrender-dev \
#     libjpeg-dev \
#     libpng-dev \
#     && rm -rf /var/lib/apt/lists/*

# # Stage 2: Builder
# FROM base as builder
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     pkg-config \
#     && rm -rf /var/lib/apt/lists/*

# # JetPack-specific PyTorch installation
# ARG JETPACK_VERSION
# RUN if [ "$JETPACK_VERSION" = "r32.7.1" ]; then \
#         pip3 install --user torch==1.9.0 torchvision==0.10.0 --extra-index-url https://download.pytorch.org/whl/cu111; \
#     elif [ "$JETPACK_VERSION" = "r35.3.1" ]; then \
#         pip3 install --user torch==2.0.0 torchvision==0.15.1 --extra-index-url https://download.pytorch.org/whl/cu118; \
#     else \
#         pip3 install --user torch torchvision; \
#     fi

# COPY requirements.txt .
# RUN pip3 install --user --no-cache-dir --upgrade pip && \
#     pip3 install --user --no-cache-dir -r requirements.txt

# # Stage 3: Runtime
# FROM base as runtime
# COPY --from=builder /root/.local /root/.local
# COPY . /app
# COPY node_exporter /usr/local/bin/node_exporter
# COPY exporter.py .

# RUN mkdir -p /app/process_video \
#     /app/feedback/images \
#     /app/feedback/labels \
#     && chmod +x /usr/local/bin/node_exporter

# EXPOSE 8501 8000 9100

# HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
#     CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# CMD ["sh", "-c", "/usr/local/bin/node_exporter --web.listen-address=0.0.0.0:9100 & \
#            python3 exporter.py & \
#            streamlit run final_test.py --server.port=8501 --server.address=0.0.0.0"]

# # Stage 1: Base system setup
# FROM nvcr.io/nvidia/l4t-base:r32.7.1 as base
# ENV DEBIAN_FRONTEND=noninteractive \
#     PYTHONUNBUFFERED=1 \
#     TZ=Etc/UTC \
#     PATH="/root/.local/bin:$PATH"
    
# # Install core system dependencies
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     python3-pip \
#     python3-dev \
#     libgl1-mesa-glx \
#     libglib2.0-0 \
#     libsm6 \
#     libxext6 \
#     libxrender-dev \
#     libjpeg-dev \
#     libpng-dev \
#     && rm -rf /var/lib/apt/lists/*

# # Stage 2: Builder for Python dependencies
# FROM base as builder
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     pkg-config \
#     && rm -rf /var/lib/apt/lists/*

# # Install Python build dependencies first
# COPY requirements.txt .
# RUN pip3 install --user --no-cache-dir --upgrade pip && \
#     pip3 install --user --no-cache-dir -r requirements.txt

# # Stage 3: Runtime image
# FROM base as runtime

# # Copy Python dependencies from builder
# COPY --from=builder /root/.local /root/.local

# # Copy application files
# COPY . /app
# COPY node_exporter /usr/local/bin/node_exporter
# COPY exporter.py .

# # Create directories and set permissions
# RUN mkdir -p /app/process_video \
#     /app/feedback/images \
#     /app/feedback/labels \
#     && chmod +x /usr/local/bin/node_exporter

# # Expose ports: Streamlit, custom exporter, node_exporter
# EXPOSE 8501 8000 9100

# # Healthcheck
# HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
#     CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# # Startup command
# CMD sh -c "/usr/local/bin/node_exporter --web.listen-address=0.0.0.0:9100 & \
#            python3 exporter.py & \
#            streamlit run final_app.py --server.port=8501 --server.address=0.0.0.0"
