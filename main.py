"""
main.py

This is the file you actually run. It ties everything together:
  1. Loads and splits the data
  2. Trains the TF-IDF baseline
  3. (Optionally) fine-tunes DistilBERT
  4. Evaluates both and saves all outputs

Usage:
    python main.py                 # run both models
    python main.py --baseline-only # just the fast baseline, skip DistilBERT
"""

import sys
import argparse
import pandas as pd

# fixing Windows console not handling unicode properly
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# importing modules
from data_loader import load_fallacy_data, prepare_splits
from models_baseline import train_baseline, predict_baseline
from models_transformer import train_distilbert, predict_distilbert
from evaluation_utils import evaluate_model, run_comparative_analysis, ensure_output_dir, OUTPUT_DIR


def main():
    # set up command line arguments
    parser = argparse.ArgumentParser(
        description="Logical Fallacy Detection - Train & Evaluate"
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Only train the TF-IDF baseline (much faster, skips DistilBERT)",
    )
    args = parser.parse_args()

    # Ensuring that the output folder exists
    ensure_output_dir()

    # loading the dataset and splitting it
    texts, labels = load_fallacy_data()
    train_texts, test_texts, train_labels, test_labels, label_encoder = prepare_splits(
        texts, labels
    )
    label_names = list(label_encoder.classes_)
    num_labels  = len(label_names)

    # collecting each model's metrics here to compare later
    all_metrics = []

    # training and evaluating the baseline
    vectorizer, classifier = train_baseline(train_texts, train_labels)
    baseline_preds = predict_baseline(vectorizer, classifier, test_texts)
    baseline_metrics = evaluate_model(
        model_name="baseline",
        test_texts=test_texts,
        test_labels=test_labels,
        predictions=baseline_preds,
        label_names=label_names,
    )
    all_metrics.append(baseline_metrics)

    # training and evaluating DistilBERT (unless the user chose baseline only)
    distilbert_preds = None

    if args.baseline_only:
        print("\n --baseline-only flag set. Skipping DistilBERT training.")
    else:
        trainer, tokenizer = train_distilbert(
            train_texts, train_labels,
            test_texts, test_labels,
            num_labels,
        )
        distilbert_preds = predict_distilbert(trainer, tokenizer, test_texts)
        distilbert_metrics = evaluate_model(
            model_name="distilbert",
            test_texts=test_texts,
            test_labels=test_labels,
            predictions=distilbert_preds,
            label_names=label_names,
        ) 
        all_metrics.append(distilbert_metrics)

        # side-by-side comparison (if both ran)
        run_comparative_analysis(
            test_texts, test_labels,
            baseline_preds, distilbert_preds,
            label_names,
        )

    # saving the model comparison table
    comparison_df = pd.DataFrame(all_metrics)
    comparison_path = OUTPUT_DIR / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\n[SAVED] Model comparison -> {comparison_path}")

    # Printing Summary:
    print(" Results summary:")
    print(comparison_df.to_string(index=False))
    print(f"\n  All output files saved in: {OUTPUT_DIR.resolve()}")

    # listing all the files that were generated
    print("\n  Generated files:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            print(f"{f.name} ({size_kb:.1f} KB)")
    print()


if __name__ == "__main__":
    main()
