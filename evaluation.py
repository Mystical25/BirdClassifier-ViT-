import numpy as np

from datasets import load_dataset

from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    Trainer
)

import evaluate


# ==========================================
# CONFIG
# ==========================================

MODEL_PATH = "models/bird_vit"


# ==========================================
# LOAD DATASET
# ==========================================

dataset = load_dataset(
    "bentrevett/caltech-ucsd-birds-200-2011"
)

print("Dataset Loaded")
print(dataset)

# ==========================================
# LOAD PROCESSOR + MODEL
# ==========================================

processor = ViTImageProcessor.from_pretrained(
    MODEL_PATH
)

model = ViTForImageClassification.from_pretrained(
    MODEL_PATH
)

# ==========================================
# METRIC
# ==========================================

metric = evaluate.load(
    "accuracy"
)

# ==========================================
# PREPROCESS FUNCTION
# ==========================================

def preprocess(example):

    processed = processor(
        images=example["image"],
        return_tensors="pt"
    )

    return {
        "pixel_values": processed["pixel_values"][0],
        "labels": example["label"]
    }


# ==========================================
# PROCESS TEST DATASET
# ==========================================

print("Preprocessing dataset...")

test_dataset = dataset["test"].map(
    preprocess
)

test_dataset.set_format(
    type="torch",
    columns=[
        "pixel_values",
        "labels"
    ]
)

print("Dataset ready")

# ==========================================
# CREATE TRAINER
# ==========================================

trainer = Trainer(
    model=model
)

# ==========================================
# RUN PREDICTIONS
# ==========================================

print("Running evaluation...")

predictions = trainer.predict(
    test_dataset
)

# ==========================================
# COMPUTE ACCURACY
# ==========================================

preds = np.argmax(
    predictions.predictions,
    axis=1
)

score = metric.compute(
    predictions=preds,
    references=predictions.label_ids
)

print("\n====================")
print("RESULTS")
print("====================")
print(
    f"Accuracy: {score['accuracy']:.4f}"
)