"""Train and evaluate the first leak-resistant purchase-intent baseline."""

import argparse
import json
from pathlib import Path
import sys
import warnings

import joblib
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Allow imports from the project and scripts directories when run from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from build_purchase_intent_dataset import FEATURE_COLUMNS, METADATA_COLUMNS, TARGET_COLUMN


DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "purchase_intent_training.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "purchase_intent_baseline.joblib"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "models" / "purchase_intent_baseline_metrics.json"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a Logistic Regression purchase-intent baseline.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the user holdout and model.")
    parser.add_argument(
        "--test-snapshot-count",
        type=int,
        default=5,
        help="Number of final prediction snapshots reserved for testing (default: 5).",
    )
    return parser.parse_args()


def load_dataset(dataset_path):
    """Load the CSV and confirm it matches the feature-engineering contract."""
    dataset = pd.read_csv(dataset_path, parse_dates=["prediction_timestamp"])
    required_columns = set(METADATA_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN])
    missing_columns = required_columns.difference(dataset.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")
    return dataset


def split_by_time_and_user(dataset, test_snapshot_count, seed):
    """Create a strict future-time test set with an entirely held-out user cohort."""
    timestamps = sorted(dataset["prediction_timestamp"].unique())
    if len(timestamps) <= test_snapshot_count:
        raise ValueError("The dataset needs more snapshots than the reserved test-snapshot count.")

    training_timestamps = timestamps[:-test_snapshot_count]
    test_timestamps = timestamps[-test_snapshot_count:]
    test_period = dataset.loc[dataset["prediction_timestamp"].isin(test_timestamps)]

    # Stratify the user cohort by whether the user has any positive row in the future
    # test period. This keeps the rare positive class represented reproducibly.
    all_user_ids = sorted(dataset["user_id"].unique())
    user_test_labels = (
        test_period.groupby("user_id")[TARGET_COLUMN].max().reindex(all_user_ids, fill_value=0)
    )
    _, held_out_user_ids = train_test_split(
        all_user_ids,
        test_size=0.20,
        random_state=seed,
        stratify=user_test_labels,
    )
    held_out_user_ids = set(held_out_user_ids)

    # Strictly exclude test users from all training rows and train-period users from
    # all test rows. No user and no future snapshot is shared between the two sets.
    train_rows = dataset.loc[
        dataset["prediction_timestamp"].isin(training_timestamps)
        & ~dataset["user_id"].isin(held_out_user_ids)
    ].copy()
    test_rows = dataset.loc[
        dataset["prediction_timestamp"].isin(test_timestamps)
        & dataset["user_id"].isin(held_out_user_ids)
    ].copy()

    if train_rows.empty or test_rows.empty:
        raise RuntimeError("The requested split produced an empty train or test set.")

    return train_rows, test_rows, training_timestamps, test_timestamps, held_out_user_ids


def build_pipeline(seed):
    """Return a single reusable preprocessing and Logistic Regression pipeline."""
    return Pipeline(
        steps=[
            # In this dataset, a missing historical purchase count means no prior purchase.
            # The same zero fallback also keeps the pipeline safe for future sparse inputs.
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    random_state=seed,
                    max_iter=1000,
                ),
            ),
        ]
    )


def evaluate(pipeline, test_rows):
    """Compute baseline metrics at the default 0.5 probability threshold."""
    y_true = test_rows[TARGET_COLUMN]
    predicted_probability = pipeline.predict_proba(test_rows[FEATURE_COLUMNS])[:, 1]
    y_predicted = (predicted_probability >= 0.5).astype(int)
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_true, y_predicted, average="binary", zero_division=0
    )
    matrix = confusion_matrix(y_true, y_predicted, labels=[0, 1])

    return {
        "accuracy": accuracy_score(y_true, y_predicted),
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "roc_auc": roc_auc_score(y_true, predicted_probability),
        "pr_auc": average_precision_score(y_true, predicted_probability),
        "positive_class_support": int(y_true.sum()),
        "test_row_count": int(len(y_true)),
        "confusion_matrix": matrix.tolist(),
    }


def main():
    args = parse_arguments()
    if args.test_snapshot_count < 1:
        raise ValueError("--test-snapshot-count must be at least 1.")

    dataset = load_dataset(args.dataset)
    train_rows, test_rows, training_timestamps, test_timestamps, held_out_user_ids = split_by_time_and_user(
        dataset, args.test_snapshot_count, args.seed
    )
    pipeline = build_pipeline(args.seed)

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", ConvergenceWarning)
        pipeline.fit(train_rows[FEATURE_COLUMNS], train_rows[TARGET_COLUMN])

    metrics = evaluate(pipeline, test_rows)
    metrics["training_row_count"] = int(len(train_rows))
    metrics["training_positive_count"] = int(train_rows[TARGET_COLUMN].sum())
    metrics["test_positive_rate"] = float(test_rows[TARGET_COLUMN].mean())
    metrics["held_out_user_count"] = len(held_out_user_ids)
    metrics["training_snapshots"] = [str(timestamp) for timestamp in training_timestamps]
    metrics["test_snapshots"] = [str(timestamp) for timestamp in test_timestamps]
    metrics["feature_columns"] = FEATURE_COLUMNS
    metrics["warnings"] = [str(warning.message) for warning in caught_warnings]

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"pipeline": pipeline, "feature_columns": FEATURE_COLUMNS, "target_column": TARGET_COLUMN},
        args.model_output,
    )
    args.metrics_output.write_text(json.dumps(metrics, indent=2) + "\n")

    print(f"Training rows: {metrics['training_row_count']} ({metrics['training_positive_count']} positives)")
    print(f"Test rows: {metrics['test_row_count']} ({metrics['positive_class_support']} positives)")
    print(f"Held-out users: {metrics['held_out_user_count']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-score: {metrics['f1_score']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"PR-AUC: {metrics['pr_auc']:.4f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(metrics["confusion_matrix"])
    print(f"Saved model pipeline: {args.model_output}")
    print(f"Saved metrics: {args.metrics_output}")
    if metrics["warnings"]:
        print("Warnings:")
        for warning in metrics["warnings"]:
            print(f"- {warning}")


if __name__ == "__main__":
    main()
