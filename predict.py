import torch

from PIL import Image

from transformers import (
    ViTImageProcessor,
    ViTForImageClassification
)


MODEL_PATH = "models/bird_vit"


processor = ViTImageProcessor.from_pretrained(
    MODEL_PATH
)

model = ViTForImageClassification.from_pretrained(
    MODEL_PATH
)

model.eval()


image = Image.open(
    "test1.jpg"
).convert("RGB")


inputs = processor(
    image,
    return_tensors="pt"
)


with torch.no_grad():

    outputs = model(
        **inputs
    )

    prediction = torch.argmax(
        outputs.logits,
        dim=1
    ).item()


label = model.config.id2label[
    prediction
]

print(
    "\nPrediction:",
    label
)