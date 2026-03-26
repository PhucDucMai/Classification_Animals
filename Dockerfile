FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Runtime libraries needed by opencv-python.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1 torchvision==0.20.1 \
    && pip install -r /app/requirements.txt

COPY . /app
WORKDIR /app/DL_CV/data

# Default command can be overridden by docker-compose or docker run.
CMD ["python", "train.py"]
