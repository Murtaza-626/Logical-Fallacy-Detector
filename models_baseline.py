"""
models_baseline.py

Contains the baseline model: TF-IDF vectorizer + Logistic Regression.
This is the simpler, faster model we use as a comparison point
against the transformer.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def train_baseline(train_texts, train_labels):
    """
    Trains a TF-IDF + Logistic Regression pipeline.

    Args:
        train_texts:  list of training argument strings
        train_labels: numpy array of integer labels

    Returns:
        vectorizer: the fitted TF-IDF vectorizer
        classifier: the fitted logistic regression model
    """
    print("Training: TF-IDF + Logistic Regression (Baseline)")

    # turn text into TF-IDF features
    # using unigrams + bigrams (single words and two-word combos)
    # capping at 10k features so the model doesn't get too big
    # sublinear_tf applies log scaling which tends to help
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    X_train = vectorizer.fit_transform(train_texts)

    # logistic regression for multi-class classification
    # balanced class weights help because some fallacy types have way fewer examples
    # max_iter=1000 so it has enough iterations to converge
    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )
    classifier.fit(X_train, train_labels)

    print("[INFO] Baseline model trained successfully.")
    return vectorizer, classifier


def predict_baseline(vectorizer, classifier, test_texts):
    """Runs the baseline model on new texts and returns predictions."""
    X_test = vectorizer.transform(test_texts)
    predictions = classifier.predict(X_test)
    return predictions
