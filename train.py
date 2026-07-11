import numpy as np

from datasets import load_dataset

from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    TrainingArguments,
    Trainer
)

import evaluate


MODEL_NAME = "google/vit-base-patch16-224"

OUTPUT_DIR = "models/bird_vit"

BATCH_SIZE = 16

EPOCHS = 5




dataset = load_dataset("bentrevett/caltech-ucsd-birds-200-2011"
)

print(dataset)




labels = dataset["train"].features[
    "label"
].names

num_labels = len(labels)

label2id = {
    label: str(i)
    for i, label in enumerate(labels)
}

id2label = {
    str(i): label
    for i, label in enumerate(labels)
}




processor = ViTImageProcessor.from_pretrained(
    MODEL_NAME
)



def transform(example_batch):

    images = [
        image.convert("RGB")
        for image in example_batch["image"]
    ]

    inputs = processor(
        images,
        return_tensors="pt"
    )

    inputs["labels"] = (
        example_batch["label"]
    )

    return inputs


prepared_dataset = dataset.with_transform(
    transform
)



model = ViTForImageClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)



accuracy = evaluate.load(
    "accuracy"
)


def compute_metrics(eval_pred):

    predictions, labels = eval_pred

    predictions = np.argmax(
        predictions,
        axis=1
    )

    return accuracy.compute(
        predictions=predictions,
        references=labels
    )




training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    per_device_train_batch_size=32,

    per_device_eval_batch_size=32,

    num_train_epochs=EPOCHS,

    eval_strategy="epoch",

    save_strategy="epoch",

    logging_steps=50,

    load_best_model_at_end=True,

    remove_unused_columns=False,

    fp16=True
)



trainer = Trainer(
    model=model,

    args=training_args,

    train_dataset=prepared_dataset[
        "train"
    ],

    eval_dataset=prepared_dataset[
        "test"
    ],

    compute_metrics=compute_metrics
)



trainer.train()



trainer.save_model(
    OUTPUT_DIR
)

processor.save_pretrained(
    OUTPUT_DIR
)

print(
    "Model saved successfully"
)