"""Build a leak-free, user-level training dataset for 7-day purchase intent.

Each row represents a user at a historical prediction timestamp. Features use
only events at or before that timestamp; the label looks only at the following
seven days.
"""

import argparse
from pathlib import Path
import sys

import pandas as pd

# Allow this script to import the app package when run from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.db.connection import get_db_connection


DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "purchase_intent_training.csv"
# Keep identifiers and time only for joining, auditing, and time-aware splitting.
# They are deliberately excluded from the ML model input matrix.
METADATA_COLUMNS = ["user_id", "prediction_timestamp"]
TARGET_COLUMN = "purchased_within_7_days"
FEATURE_EVENT_TYPES = ["search", "view", "wishlist", "cart"]
FEATURE_COLUMNS = [
    "search_count",
    "view_count",
    "wishlist_count",
    "cart_count",
    "purchase_count_before_prediction",
    "total_event_count",
    "activity_days",
    "recency_hours",
    "distinct_products_viewed",
    "distinct_categories_interacted",
    "average_interacted_product_price",
    "average_carted_product_price",
]


def parse_arguments():
    parser = argparse.ArgumentParser(description="Build a 7-day purchase-intent training dataset.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"CSV output path (default: {DEFAULT_OUTPUT_PATH.relative_to(PROJECT_ROOT)}).",
    )
    parser.add_argument(
        "--snapshot-interval-days",
        type=int,
        default=7,
        help="Days between global prediction snapshots (default: 7).",
    )
    parser.add_argument(
        "--min-history-events",
        type=int,
        default=2,
        help="Minimum events required before a user appears at a snapshot (default: 2).",
    )
    return parser.parse_args()


def load_events():
    """Load event history with the product properties needed for feature calculations."""
    query = """
        SELECT
            e.user_id,
            e.product_id,
            e.event_type,
            e.event_time,
            p.category,
            p.price::double precision AS price
        FROM events AS e
        LEFT JOIN products AS p ON p.product_id = e.product_id
        WHERE e.user_id IS NOT NULL
          AND e.event_time IS NOT NULL
        ORDER BY e.event_time, e.event_id
    """
    with get_db_connection() as connection:
        events = pd.read_sql_query(query, connection, parse_dates=["event_time"])

    if events.empty:
        raise RuntimeError("No usable events found. Generate synthetic events before building the dataset.")
    return events


def prediction_timestamps(events, interval_days):
    """Return snapshots whose full future target windows are observable in the data."""
    first_day = events["event_time"].min().normalize()
    last_valid_day = (events["event_time"].max() - pd.Timedelta(days=7)).normalize()
    if last_valid_day < first_day:
        raise RuntimeError("At least seven days of event history are required to create a target window.")
    return pd.date_range(first_day, last_valid_day, freq=f"{interval_days}D")


def build_snapshot_rows(events, prediction_timestamp, min_history_events):
    """Create features and a target for every eligible user at one timestamp."""
    history = events.loc[events["event_time"] <= prediction_timestamp].copy()
    history_sizes = history.groupby("user_id").size()
    eligible_user_ids = history_sizes[history_sizes >= min_history_events].index
    history = history.loc[history["user_id"].isin(eligible_user_ids)]

    if history.empty:
        return pd.DataFrame()

    # Counts are explicitly limited to event history available at this prediction time.
    event_counts = (
        history.groupby(["user_id", "event_type"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=FEATURE_EVENT_TYPES, fill_value=0)
    )
    features = event_counts.rename(
        columns={event_type: f"{event_type}_count" for event_type in FEATURE_EVENT_TYPES}
    )
    features["total_event_count"] = history.groupby("user_id").size()
    features["activity_days"] = history.groupby("user_id")["event_time"].apply(
        lambda times: times.dt.normalize().nunique()
    )
    features["recency_hours"] = (
        prediction_timestamp - history.groupby("user_id")["event_time"].max()
    ).dt.total_seconds() / 3600
    features["distinct_products_viewed"] = (
        history.loc[history["event_type"] == "view"].groupby("user_id")["product_id"].nunique()
    )
    features["distinct_categories_interacted"] = history.groupby("user_id")["category"].nunique()
    features["purchase_count_before_prediction"] = (
        history.loc[history["event_type"] == "purchase"].groupby("user_id").size()
    )
    features["average_interacted_product_price"] = history.groupby("user_id")["price"].mean()
    features["average_carted_product_price"] = (
        history.loc[history["event_type"] == "cart"].groupby("user_id")["price"].mean()
    )

    # A missing event-derived count means zero. This makes "no previous purchase"
    # explicitly 0 in purchase_count_before_prediction. Price is zero only when no
    # cart exists; cart_count remains available so a later model can distinguish it.
    count_columns = [column for column in features if column.endswith("_count")]
    features[count_columns] = features[count_columns].fillna(0).astype(int)
    features[["distinct_products_viewed", "distinct_categories_interacted"]] = features[
        ["distinct_products_viewed", "distinct_categories_interacted"]
    ].fillna(0).astype(int)
    features["average_carted_product_price"] = features["average_carted_product_price"].fillna(0.0)

    target_window_end = prediction_timestamp + pd.Timedelta(days=7)
    future_purchasers = events.loc[
        (events["event_time"] > prediction_timestamp)
        & (events["event_time"] <= target_window_end)
        & (events["event_type"] == "purchase"),
        "user_id",
    ].unique()

    features["purchased_within_7_days"] = features.index.isin(future_purchasers).astype(int)
    features = features.reset_index()
    features.insert(1, "prediction_timestamp", prediction_timestamp)
    return features[METADATA_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN]]


def main():
    args = parse_arguments()
    if args.snapshot_interval_days < 1 or args.min_history_events < 1:
        raise ValueError("--snapshot-interval-days and --min-history-events must be positive integers.")

    events = load_events()
    snapshots = prediction_timestamps(events, args.snapshot_interval_days)
    dataset_parts = [
        build_snapshot_rows(events, prediction_timestamp, args.min_history_events)
        for prediction_timestamp in snapshots
    ]
    dataset_parts = [part for part in dataset_parts if not part.empty]
    if not dataset_parts:
        raise RuntimeError("No users meet the minimum history requirement at any prediction snapshot.")
    dataset = pd.concat(dataset_parts, ignore_index=True)
    dataset = dataset.sort_values(["prediction_timestamp", "user_id"]).reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output, index=False)

    positive_rate = dataset["purchased_within_7_days"].mean()
    print(f"Wrote {len(dataset)} training rows to {args.output}")
    print(f"Prediction snapshots: {len(snapshots)}")
    print(f"Positive target rate: {positive_rate:.2%}")


if __name__ == "__main__":
    main()
