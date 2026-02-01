# Gunakan Base Image NVIDIA CUDA 12
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# Hindari interaksi saat install package
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies sistem (Python, FFmpeg, git)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    ffmpeg \
    git \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# --- [PENTING] INSTALL TORCH MANUAL ---
# Kita install Torch secara terpisah agar layer ini di-cache dan tidak perlu download ulang terus menerus.
# Kita gunakan versi CUDA 12.1 (cu121) agar kompatibel dengan driver RTX 4050 Anda.
RUN python3 -m pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu121 --break-system-packages

# Install sisa requirements (FastAPI, yt-dlp, faster-whisper)
RUN python3 -m pip install --no-cache-dir -r requirements.txt --break-system-packages

# Copy source code
COPY . .

# Environment Variable
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu

# Jalankan app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
