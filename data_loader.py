"""
data_loader.py

Handles all the data stuff:
  - Downloads the logical fallacy dataset from HuggingFace
  - Converts the string labels into numbers
  - Splits everything into train/test (80/20, stratified)
  - Falls back to a local CSV if HuggingFace doesn't work

You can run this file on its own to check the data:
    python data_loader.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_fallacy_data():
    """
    Tries to grab the dataset from HuggingFace first.
    If that doesn't work, it looks for a local 'fallacy_data.csv' file instead.

    Returns:
        texts  (list of str): the argument/statement texts
        labels (list of str): the fallacy type for each text
    """
    try:
        from datasets import load_dataset, concatenate_datasets
        print("Loading dataset from HuggingFace: tasksource/logical-fallacy")

        # the dataset has train/test/dev splits on HuggingFace,
        # but we want all the data so we can make our own split later
        all_splits = load_dataset("tasksource/logical-fallacy")
        combined = concatenate_datasets(
            [all_splits[split] for split in all_splits.keys()]
        )

        texts  = combined["source_article"]
        labels = combined["logical_fallacies"]
        print(f"Successfully loaded {len(texts)} samples from HuggingFace.")
        return texts, labels

    except Exception as e:
        print(f"HuggingFace loading failed: {e}")
        print("Trying local CSV fallback (fallacy_data.csv) ...")

        csv_path = os.path.join(os.path.dirname(__file__), "fallacy_data.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Could not load from HuggingFace and no local CSV found at:\n"
                f"  {csv_path}\n"
                f"Please download the dataset manually or check your internet connection."
            )

        df = pd.read_csv(csv_path)
        texts  = df["source_article"].tolist()
        labels = df["logical_fallacies"].tolist()
        print(f"Loaded {len(texts)} samples from local CSV.")
        return texts, labels


def prepare_splits(texts, labels, test_size=0.2, random_state=42):
    """
    Takes the raw texts and labels, encodes them as integers,
    then does a stratified 80/20 split so each fallacy type is
    represented fairly in both train and test.

    Returns:
        train_texts, test_texts, train_labels, test_labels, label_encoder
    """
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # stratified means each class keeps roughly the same proportion in both sets
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        encoded_labels,
        test_size=test_size,
        random_state=random_state,
        stratify=encoded_labels
    )

    # quick summary so you know what you're working with
    print(f"Dataset Summary")
    print(f"Total samples:		{len(texts)}")
    print(f"Training samples:	{len(train_texts)}")
    print(f"Test samples:		{len(test_texts)}")
    print(f"Number of classes:	{len(label_encoder.classes_)}")
    print(f"Classes: {list(label_encoder.classes_)}")

    return train_texts, test_texts, train_labels, test_labels, label_encoder


if __name__ == "__main__":
    texts, labels = load_fallacy_data()
    train_texts, test_texts, train_labels, test_labels, le = prepare_splits(texts, labels)

    print("Label distribution in training set:")
    unique, counts = np.unique(train_labels, return_counts=True)
    for label_id, count in zip(unique, counts):
        label_name = le.classes_[label_id]
        print(f"  {label_name:30s} -> {count:4d} samples")
