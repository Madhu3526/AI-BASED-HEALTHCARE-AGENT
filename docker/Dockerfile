# ─────────────────────────────────────────────────────────────────────────────
# Federated Medical Diagnosis Support System
# Multi-stage Docker build for reproducible deployment
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Base Python + system libraries ───────────────────────────────────
FROM python:3.10-slim AS base

# Prevent interactive prompts during package install
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System dependencies for OpenCV and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# ── Stage 2: Python dependencies ─────────────────────────────────────────────
FROM base AS deps

WORKDIR /install

COPY requirements.txt .

# Install CPU-only PyTorch first (smaller image, suitable for inference)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install -r requirements.txt

# ── Stage 3: Application ──────────────────────────────────────────────────────
FROM deps AS app

WORKDIR /app

# Copy source code
COPY . .

# Create directories for models and data
RUN mkdir -p \
    models/federated \
    data/nih_xray \
    explainability/outputs

# ── Streamlit configuration ────────────────────────────────────────────────────
RUN mkdir -p /root/.streamlit
RUN cat > /root/.streamlit/config.toml <<EOF
[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = true

[theme]
base = "dark"
primaryColor = "#4fc3f7"
backgroundColor = "#0e1117"
secondaryBackgroundColor = "#1e2329"
textColor = "#cfd8dc"
EOF

# ── Expose Streamlit port ─────────────────────────────────────────────────────
EXPOSE 8501

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ── Entry point ────────────────────────────────────────────────────────────────
CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]
