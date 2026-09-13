"""Generate realistic, local e-commerce event data for ML experimentation.

The script adds users only when the database has fewer than --target-users.
It does not change any database tables or remove existing data.
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import random
import sys

from psycopg2.extras import execute_values

# Allow this script to import the app package when run from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.db.connection import get_db_connection


PROFILE_SESSION_RANGES = {
    "casual_browser": (5, 8),
    "considering": (7, 11),
    "high_intent": (9, 14),
}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate synthetic e-commerce users and events.")
    parser.add_argument("--target-users", type=int, default=500, help="Total users to reach (default: 500).")
    parser.add_argument("--days", type=int, default=180, help="How many recent days to simulate (default: 180).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for repeatable data generation.")
    return parser.parse_args()


def random_datetime(start, end, rng):
    """Return a random datetime between start and end."""
    return start + (end - start) * rng.random()


def create_missing_users(cursor, number_to_create, now, rng):
    """Insert users with realistic historical account-creation times."""
    if number_to_create <= 0:
        return 0

    # New synthetic users predate their events, while the event history itself is recent.
    account_start = now - timedelta(days=365)
    account_end = now - timedelta(days=181)
    user_rows = [(random_datetime(account_start, account_end, rng),) for _ in range(number_to_create)]
    execute_values(cursor, "INSERT INTO users (created_at) VALUES %s", user_rows)
    return number_to_create


def choose_product(products, preferred_category, rng):
    """Favor a user's preferred category while occasionally allowing discovery."""
    preferred_products = [product for product in products if product[2] == preferred_category]
    if preferred_products and rng.random() < 0.8:
        return rng.choice(preferred_products)
    return rng.choice(products)


def build_session(profile, products, preferred_category, session_start, rng):
    """Build an ordered event sequence for one browsing session.

    A session always starts with discovery and product views. Higher-intent
    profiles are progressively more likely to wishlist, cart, and purchase.
    """
    target_product = choose_product(products, preferred_category, rng)
    events = [("search", target_product[0]), ("view", target_product[0])]

    # Comparing one or two alternatives is a common behavior before conversion.
    for _ in range(rng.randint(0, 2)):
        alternative = choose_product(products, preferred_category, rng)
        events.append(("view", alternative[0]))

    if profile == "casual_browser":
        if rng.random() < 0.15:
            events.append(("wishlist", target_product[0]))
    elif profile == "considering":
        if rng.random() < 0.60:
            events.append(("wishlist", target_product[0]))
        if rng.random() < 0.28:
            events.append(("cart", target_product[0]))
            if rng.random() < 0.12:
                events.append(("purchase", target_product[0]))
    else:  # high_intent
        if rng.random() < 0.45:
            events.append(("wishlist", target_product[0]))
        events.append(("cart", target_product[0]))
        if rng.random() < 0.70:
            events.append(("purchase", target_product[0]))

    # Events occur minutes apart and preserve the natural funnel ordering above.
    event_rows = []
    event_time = session_start
    for event_type, product_id in events:
        event_time += timedelta(minutes=rng.randint(1, 15))
        event_rows.append((product_id, event_type, event_time))
    return event_rows


def build_events(users, products, days, now, rng):
    """Create event rows for every user, assigning each a distinct behavior profile."""
    history_start = now - timedelta(days=days)
    categories = sorted({product[2] for product in products})
    event_rows = []

    for user_id, created_at in users:
        profile = rng.choices(
            ["casual_browser", "considering", "high_intent"], weights=[60, 30, 10], k=1
        )[0]
        preferred_category = rng.choice(categories)
        session_count = rng.randint(*PROFILE_SESSION_RANGES[profile])
        # created_at is nullable in the current schema, so handle legacy rows safely.
        user_history_start = max(history_start, created_at or history_start)

        # A very recently created user can still receive a small same-day history.
        if user_history_start >= now:
            user_history_start = now - timedelta(hours=2)

        for _ in range(session_count):
            session_start = random_datetime(user_history_start, now - timedelta(minutes=30), rng)
            for product_id, event_type, event_time in build_session(
                profile, products, preferred_category, session_start, rng
            ):
                event_rows.append((user_id, product_id, event_type, event_time))

    return event_rows


def main():
    args = parse_arguments()
    if args.target_users < 1 or args.days < 1:
        raise ValueError("--target-users and --days must both be positive integers.")

    rng = random.Random(args.seed)
    now = datetime.now().replace(microsecond=0)

    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM users")
            existing_user_count = cursor.fetchone()[0]
            users_to_create = max(0, args.target_users - existing_user_count)
            created_user_count = create_missing_users(cursor, users_to_create, now, rng)

            cursor.execute("SELECT user_id, created_at FROM users ORDER BY user_id")
            users = cursor.fetchall()
            cursor.execute("SELECT product_id, product_name, category, price FROM products ORDER BY product_id")
            products = cursor.fetchall()

            if not products:
                raise RuntimeError("No products found. Add products before generating events.")

            event_rows = build_events(users, products, args.days, now, rng)
            execute_values(
                cursor,
                """
                INSERT INTO events (user_id, product_id, event_type, event_time)
                VALUES %s
                """,
                event_rows,
                page_size=1000,
            )

    print(f"Created users: {created_user_count}")
    print(f"Generated events: {len(event_rows)}")
    print(f"Total users with simulated behavior: {len(users)}")


if __name__ == "__main__":
    main()
