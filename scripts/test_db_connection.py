"""Verify that the local application can connect to PostgreSQL."""

from pathlib import Path
import sys

# Allow this script to import the app package when run from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.db.connection import get_db_connection


def main():
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM users")
            user_count = cursor.fetchone()[0]

    print(f"Database connection successful. Users: {user_count}")


if __name__ == "__main__":
    main()
