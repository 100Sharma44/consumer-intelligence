"""Compare small, reproducible purchase-intent models without touching the test set."""

import argparse
import json
from pathlib import Path
import sys
import warnings

import joblib
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Reuse the baseline split implementation so test users and time windows are identical.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from build_purchase_intent_dataset import FEATURE_COLUMNS, TARGET_COLUMN
from train_purchase_intent_baseline import load_dataset, split_by_time_and_user


DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "purchase_intent_training.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "purchase_intent_tuned.joblib"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "models" / "purchase_intent_tuned_metrics.json"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Compare baseline purchase-intent model candidates.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-snapshot-count", type=int, default=5)
    parser.add_argument(
        "--validation-snapshot-count",
        type=int,
        default=4,
        help="Final training-period snapshots reserved for model and threshold selection (default: 4).",
    )
    return parser.parse_args()


def make_pipeline(classifier):
    """Keep preprocessing identical for every candidate."""
    return Pipeline(
        steps=[
            # Missing purchase history represents no historical purchase in this dataset.
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )


def candidate_pipelines(seed):
    """Return a deliberately small set of imbalance-aware model candidates."""
    return {
        "logistic_regression": make_pipeline(
            LogisticRegression(class_weight="balanced", random_state=seed, max_iter=1000)
        ),
        "random_forest": make_pipeline(
            RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=-1,
            )
        ),
        "extra_trees": make_pipeline(
            ExtraTreesClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            )
        ),
    }


def choose_f1_threshold(y_true, probabilities):
    """Choose the validation threshold with the largest positive-class F1 score."""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    best_index = f1_scores.argmax()
    return float(thresholds[best_index])


def evaluate_at_threshold(y_true, probabilities, threshold):
    """Measure ranking and threshold-dependent classification quality."""
    predictions = (probabilities >= threshold).astype(int)
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_true, predictions, average="binary", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "positive_class_support": int(y_true.sum()),
        "row_count": int(len(y_true)),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1]).tolist(),
    }


def main():
    args = parse_arguments()
    if args.test_snapshot_count < 1 or args.validation_snapshot_count < 1:
        raise ValueError("Snapshot counts must be positive integers.")

    dataset = load_dataset(args.dataset)
    train_rows, test_rows, training_timestamps, test_timestamps, held_out_user_ids = split_by_time_and_user(
        dataset, args.test_snapshot_count, args.seed
    )
    if len(training_timestamps) <= args.validation_snapshot_count:
        raise ValueError("More training snapshots are required than validation snapshots.")

    model_timestamps = training_timestamps[: -args.validation_snapshot_count]
    validation_timestamps = training_timestamps[-args.validation_snapshot_count :]
    model_rows = train_rows.loc[train_rows["prediction_timestamp"].isin(model_timestamps)]
    validation_rows = train_rows.loc[train_rows["prediction_timestamp"].isin(validation_timestamps)]

    candidate_results = {}
    selected_name = None
    selected_threshold = None
    selected_sort_key = None
    for name, pipeline in candidate_pipelines(args.seed).items():
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", ConvergenceWarning)
            pipeline.fit(model_rows[FEATURE_COLUMNS], model_rows[TARGET_COLUMN])

        validation_probabilities = pipeline.predict_proba(validation_rows[FEATURE_COLUMNS])[:, 1]
        threshold = choose_f1_threshold(validation_rows[TARGET_COLUMN], validation_probabilities)
        validation_metrics = evaluate_at_threshold(
            validation_rows[TARGET_COLUMN], validation_probabilities, threshold
        )
        candidate_results[name] = {
            "selection_threshold": threshold,
            "validation_metrics": validation_metrics,
            "warnings": [str(warning.message) for warning in caught_warnings],
        }

        # PR-AUC is primary for rare-event ranking. F1 at the validation-only threshold
        # breaks ties and selects an operating point without inspecting the final test set.
        selection_key = (validation_metrics["pr_auc"], validation_metrics["f1_score"])
        if selected_sort_key is None or selection_key > selected_sort_key:
            selected_name = name
            selected_threshold = threshold
            selected_sort_key = selection_key

    final_pipeline = candidate_pipelines(args.seed)[selected_name]
    with warnings.catch_warnings(record=True) as final_warnings:
        warnings.simplefilter("always", ConvergenceWarning)
        final_pipeline.fit(train_rows[FEATURE_COLUMNS], train_rows[TARGET_COLUMN])

    test_probabilities = final_pipeline.predict_proba(test_rows[FEATURE_COLUMNS])[:, 1]
    final_test_metrics = evaluate_at_threshold(test_rows[TARGET_COLUMN], test_probabilities, selected_threshold)

    results = {
        "selected_model": selected_name,
        "selected_threshold": selected_threshold,
        "selection_rule": "highest validation PR-AUC; validation F1 breaks ties and determines threshold",
        "candidate_validation_results": candidate_results,
        "final_test_metrics": final_test_metrics,
        "feature_columns": FEATURE_COLUMNS,
        "seed": args.seed,
        "held_out_user_count": len(held_out_user_ids),
        "training_row_count": int(len(train_rows)),
        "training_positive_count": int(train_rows[TARGET_COLUMN].sum()),
        "model_selection_row_count": int(len(model_rows)),
        "validation_row_count": int(len(validation_rows)),
        "test_row_count": int(len(test_rows)),
        "training_snapshots": [str(timestamp) for timestamp in training_timestamps],
        "validation_snapshots": [str(timestamp) for timestamp in validation_timestamps],
        "test_snapshots": [str(timestamp) for timestamp in test_timestamps],
        "final_fit_warnings": [str(warning.message) for warning in final_warnings],
    }

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": final_pipeline,
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
            "classification_threshold": selected_threshold,
        },
        args.model_output,
    )
    args.metrics_output.write_text(json.dumps(results, indent=2) + "\n")

    print(f"Selected model: {selected_name}")
    print(f"Selected threshold: {selected_threshold:.4f}")
    print("Validation comparison (PR-AUC, F1, threshold):")
    for name, result in candidate_results.items():
        metrics = result["validation_metrics"]
        print(f"- {name}: {metrics['pr_auc']:.4f}, {metrics['f1_score']:.4f}, {result['selection_threshold']:.4f}")
    print("Final test metrics:")
    for name in ("precision", "recall", "f1_score", "roc_auc", "pr_auc"):
        print(f"{name}: {final_test_metrics[name]:.4f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(final_test_metrics["confusion_matrix"])
    print(f"Saved tuned model: {args.model_output}")
    print(f"Saved tuned metrics: {args.metrics_output}")


if __name__ == "__main__":
    main()
