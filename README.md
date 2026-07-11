# Bird Species Classification using Vision Transformer (ViT)

A deep learning project that fine-tunes a **Vision Transformer (ViT)** for fine-grained bird species classification on the **Caltech-UCSD Birds-200-2011 (CUB-200-2011)** dataset using the Hugging Face Transformers library.

## Features

- Fine-tuned Vision Transformer (ViT)
- Transfer learning using a pre-trained Google ViT model
- Classification of 200 bird species
- Training, evaluation, and prediction scripts
- Model saving and loading
- Custom image prediction

## Dataset

**Caltech-UCSD Birds-200-2011 (CUB-200-2011)**

- 200 bird species
- 11,788 bird images
- Fine-grained image classification benchmark

The dataset is automatically downloaded using the Hugging Face `datasets` library.

## Model

- **Base Model:** google/vit-base-patch16-224
- **Framework:** PyTorch + Hugging Face Transformers
- **Task:** Multi-class Image Classification

## Results

| Metric | Score |
|---------|-------|
| Test Accuracy | **85.31%** |

## Project Structure

```
BirdClassifier/
│
├── train.py          # Train the Vision Transformer
├── evaluation.py     # Evaluate the trained model
├── predict.py        # Predict bird species from an image
├── requirements.txt
├── README.md
│
└── models/
    └── bird_vit/     # Saved trained model
```

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/BirdClassifier.git
cd BirdClassifier
```

Create a virtual environment

```bash
python -m venv venv
```

Activate it

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

## Training

```bash
python train.py
```

The trained model is saved inside:

```
models/bird_vit
```

## Evaluation

```bash
python evaluation.py
```

## Prediction

Place your test image in the project directory (or update the image path inside `predict.py`) and run:

```bash
python predict.py
```

Example output

```
Prediction: Black_Footed_Albatross
```

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- NumPy
- Pillow

## Future Improvements

- Data augmentation
- Hyperparameter tuning
- Grad-CAM visualizations
- Web interface for predictions
- Deployment as a REST API

## Author

**Rigzin Gyalpo**
