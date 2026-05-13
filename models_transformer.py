"""
models_transformer.py

Contains the DistilBERT transformer model.
We fine-tune a pre-trained distilbert-base-uncased for our fallacy
classification task using the HuggingFace Trainer API.
"""

import numpy as np
from pathlib import Path
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset

# folder to store checkpoints and logs
OUTPUT_DIR = Path(__file__).parent / "outputs"


def train_distilbert(train_texts, train_labels, test_texts, test_labels, num_labels):
    """
    Fine-tunes DistilBERT on the fallacy dataset.

    Args:
        train_texts:  list of training strings
        train_labels: numpy array of integer labels for training
        test_texts:   list of test strings (used for eval during training)
        test_labels:  numpy array of integer labels for test
        num_labels:   how many unique fallacy classes there are

    Returns:
        trainer:   the trained HuggingFace Trainer (needed for prediction)
        tokenizer: the tokenizer (also needed for prediction)
    """
    print("Training: DistilBERT (Fine-tuned Transformer)")

    from transformers import (
        DistilBertTokenizerFast,
        DistilBertForSequenceClassification,
        Trainer,
        TrainingArguments,
    )

    print("Training on CPU which will take some time")

    # load the pre-trained tokenizer
    model_name = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

    # tokenize everything, truncate/pad to 256 tokens (most arguments are pretty short)
    print("Tokenizing texts ...")
    train_encodings = tokenizer(
        list(train_texts), truncation=True, padding=True, max_length=256
    )
    test_encodings = tokenizer(
        list(test_texts), truncation=True, padding=True, max_length=256
    )

    # simple wrapper so PyTorch can iterate over our tokenized data
    # 1. Add labels to the training encodings dictionary
    train_encodings["labels"] = train_labels.tolist()
    train_dataset = Dataset.from_dict(train_encodings)

    # 2. Add labels to the test encodings dictionary
    test_encodings["labels"] = test_labels.tolist()
    test_dataset = Dataset.from_dict(test_encodings)

    # load the pre-trained model with a classification head on top
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    # training settings
    # 3 epochs is pretty standard for fine-tuning
    # batch size 16 should work on most machines (try 8 if you run out of memory)
    # 2e-5 learning rate is the go-to for BERT-family models
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "distilbert_checkpoints"),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=str(OUTPUT_DIR / "logs"),
        logging_steps=50,
        report_to="none",
        seed=42,
    )

    # this function gets called at the end of each epoch to compute metrics
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="macro", zero_division=0
        )
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    # set up the trainer and start fine-tuning
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    print("Starting DistilBERT fine-tuning")
    trainer.train()
    print("DistilBERT training complete!")

    return trainer, tokenizer


def predict_distilbert(trainer, tokenizer, test_texts):
    """Runs the fine-tuned DistilBERT on new texts and returns predictions."""

    test_encodings = tokenizer(
        list(test_texts), truncation=True, padding=True, max_length=256
    )

    test_dataset = Dataset.from_dict(test_encodings)

    # trainer.predict handles batching for us automatically
    raw_preds = trainer.predict(test_dataset)
    predictions = np.argmax(raw_preds.predictions, axis=-1)
    return predictions
