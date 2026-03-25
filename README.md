# Animal Species Classification with PyTorch

This repository contains a complete deep learning pipeline to classify 10 animal species from images:

- butterfly
- cat
- chicken
- cow
- dog
- elephant
- horse
- sheep
- spider
- squirrel

The project includes:

- A custom CNN architecture
- A dataset loader for folder-based train/test images
- A training script with checkpointing and TensorBoard logging
- An inference script for single-image prediction

## Project Structure

```text
Classification_Animals/
├── DL_CV/
│   └── data/
│       ├── CNN_Model.py
│       ├── dataset_animals.py
│       ├── train.py
│       ├── test.py
│       ├── animals/
│       │   ├── train/
│       │   │   ├── butterfly/
│       │   │   ├── cat/
│       │   │   └── ...
│       │   └── test/
│       │       ├── butterfly/
│       │       ├── cat/
│       │       └── ...
│       ├── image_test/
│       ├── tensorboard/
│       └── trained_models/
└── README.md
```

## Requirements

Use Python 3.8+ (recommended: Python 3.9 or 3.10).

Install dependencies:

```bash
pip install torch torchvision numpy opencv-python scikit-learn matplotlib tqdm tensorboard
```

## Quick Start

### 1. Move to the working directory

Most paths in the code are relative to `DL_CV/data`, so run commands from there:

```bash
cd DL_CV/data
```

### 2. Verify dataset layout

The training script expects this exact structure:

```text
animals/
├── train/
│   ├── butterfly/
│   ├── cat/
│   └── ...
└── test/
    ├── butterfly/
    ├── cat/
    └── ...
```

Each class folder should contain image files.

## Train the Model

Basic training:

```bash
python train.py
```

Useful options:

```bash
python train.py \
  --batch_size 16 \
  --epoch 100 \
  --image_size 224 \
  --trained_path trained_models
```

### Resume training from checkpoint

```bash
python train.py --checkpoint_path trained_models/last.pt
```

## Outputs

After/during training, the script creates:

- `trained_models/last.pt`: checkpoint from the latest epoch
- `trained_models/best.pt`: best checkpoint by validation accuracy
- `tensorboard/`: TensorBoard logs (loss, accuracy, confusion matrix)

Note:

- `tensorboard/` is deleted and recreated each training run.

## Monitor Training with TensorBoard

Run from `DL_CV/data`:

```bash
tensorboard --logdir tensorboard --port 6006
```

Open in browser:

```text
http://localhost:6006
```

## Run Inference on One Image

Default inference (uses `trained_models/best.pt` and `image_test/cat1.jpeg`):

```bash
python test.py
```

Custom image/checkpoint:

```bash
python test.py \
  --checkpoint_path trained_models/best.pt \
  --image_path image_test/your_image.jpg \
  --image_size 224
```

Expected output example:

```text
This image about cat with probability of 0.9821
```

## Script Arguments

### train.py

- `--data_path`: dataset root path (default: `./animals`)
- `--batch_size`: batch size (default: `16`)
- `--epoch`: number of epochs (default: `100`)
- `--image_size`: input image size (default: `224`)
- `--checkpoint_path`: path to checkpoint to resume from (default: `None`)
- `--trained_path`: output directory for checkpoints (default: `trained_models`)

### test.py

- `--data_path`: dataset path (not used during current inference flow)
- `--image_size`: input size for test image (default: `224`)
- `--checkpoint_path`: checkpoint path (default: `./trained_models/best.pt`)
- `--image_path`: image to classify (default: `./image_test/cat1.jpeg`)

## How the Pipeline Works

1. `dataset_animals.py`
- Reads folder names in `animals/train` or `animals/test` as class labels.
- Loads images with OpenCV and converts them to tensors.

2. `CNN_Model.py`
- Defines a 5-block CNN with BatchNorm, LeakyReLU, and MaxPool.
- Uses fully connected layers with dropout for classification.

3. `train.py`
- Creates train/validation dataloaders.
- Trains with CrossEntropyLoss + Adam optimizer.
- Applies learning-rate scheduling (MultiStepLR).
- Logs metrics and confusion matrix to TensorBoard.
- Saves `last.pt` and `best.pt` checkpoints.

4. `test.py`
- Loads a trained checkpoint.
- Preprocesses one image.
- Runs forward pass and prints class prediction with probability.

## Troubleshooting

1. Error: checkpoint file does not exist
- Confirm path passed to `--checkpoint_path`.
- Run training first to generate `trained_models/best.pt`.

2. Error: image cannot be loaded
- Confirm `--image_path` exists.
- Verify the image is readable and has a valid extension.

3. Path-related errors
- Run scripts from `DL_CV/data` because code uses relative paths like `./animals`.

4. DataLoader worker issues on some systems
- If needed, reduce `num_workers` in `train.py` from `6` to `0` or `2`.

## Notes

- Class ordering is determined by folder listing order. Keep train/test class folder names consistent.
- For reproducibility and more stable label mapping, sorting class names in the dataset loader is a good next improvement.

## License

Add your preferred license file (for example, MIT) if you plan to share this project publicly.
