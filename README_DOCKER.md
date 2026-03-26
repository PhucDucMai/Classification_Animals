# Docker Deployment Guide

This guide explains how to run the Animal Species Classification pipeline using Docker.

## Files Added for Docker

- `Dockerfile`: builds a Python environment with all required dependencies.
- `docker-compose.yml`: defines services for training, inference, and TensorBoard.
- `.dockerignore`: excludes unnecessary files from Docker build context.
- `requirements.txt`: Python dependencies used by the image.

## Prerequisites

1. Docker Engine installed
2. Docker Compose plugin installed (`docker compose`)

Check installation:

```bash
docker --version
docker compose version
```

## Build the Docker Image

From repository root:

```bash
docker build -t animal-classifier:latest .
```

## Run Training

### Option A: Docker Compose

```bash
docker compose run --rm train
```

This project is configured with `gpus: all` for the `train` service, so if your host has NVIDIA Container Toolkit configured, training will use GPU automatically.

Quick GPU check inside container:

```bash
docker compose run --rm train nvidia-smi
```

### Option B: Docker Run

```bash
docker run --rm \
  --shm-size=4g \
  -v "$(pwd)/DL_CV/data:/app/DL_CV/data" \
  -w /app/DL_CV/data \
  animal-classifier:latest \
  python train.py
```

### Custom training arguments

```bash
docker compose run --rm train \
  python train.py --batch_size 16 --epoch 100 --image_size 224
```

## Resume Training from Checkpoint

```bash
docker compose run --rm train \
  python train.py --checkpoint_path trained_models/last.pt
```

## Run Inference

### Default test command

```bash
docker compose run --rm test
```

### Custom image/checkpoint

```bash
docker compose run --rm test \
  python test.py --checkpoint_path trained_models/best.pt --image_path image_test/cat1.jpeg
```

## Run TensorBoard in Docker

Start TensorBoard service:

```bash
docker compose up tensorboard
```

Then open:

```text
http://localhost:6006
```

Stop service:

```bash
docker compose down
```

## Data and Model Persistence

`docker-compose.yml` mounts your local folder:

- `./DL_CV/data` -> `/app/DL_CV/data`

This means checkpoints and logs remain on your host machine in:

- `DL_CV/data/trained_models`
- `DL_CV/data/tensorboard`

## Optional GPU Training

If you have NVIDIA Docker runtime configured, you can run with GPU:

```bash
docker run --rm --gpus all \
  --shm-size=4g \
  -v "$(pwd)/DL_CV/data:/app/DL_CV/data" \
  -w /app/DL_CV/data \
  animal-classifier:latest \
  python train.py
```

During training, the script now prints either:

- `Using GPU: <gpu-name>`
- or `CUDA not available. Training will run on CPU.`

So you can confirm device selection directly in logs.

## Metrics After Training Finishes

At the end of training, the script prints a summary and writes:

- `trained_models/metrics_summary.json`

This JSON includes:

- best validation accuracy and epoch
- last epoch loss/accuracy/precision/recall/F1 (macro)
- full epoch history for train loss and validation metrics

## Common Docker Commands

Rebuild image after dependency/code changes:

```bash
docker compose build --no-cache
```

Clean stopped containers, dangling images, and unused networks:

```bash
docker system prune
```

## Troubleshooting

1. `ModuleNotFoundError` inside container
- Rebuild image: `docker compose build --no-cache`

2. OpenCV runtime error (`libGL.so.1` not found)
- The Dockerfile already installs `libgl1`; rebuild the image if this occurs.

3. TensorBoard page not reachable
- Verify port mapping `6006:6006` is not occupied.
- Check service logs: `docker compose logs tensorboard`

4. Permission issues on generated files
- Files created by container may be owned by root depending on Docker settings.

## Recommended Workflow

1. Build image once.
2. Train with `docker compose run --rm train`.
3. Monitor logs with `docker compose up tensorboard`.
4. Run inference with `docker compose run --rm test`.
