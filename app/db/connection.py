"""PostgreSQL connection helpers for the consumer-intelligence application."""

import os

import psycopg2


def get_db_connection():
    """Create and return a PostgreSQL connection from environment settings."""
    required_variables = ("DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD")
    missing_variables = [name for name in required_variables if not os.getenv(name)]

    if missing_variables:
        missing = ", ".join(missing_variables)
        raise RuntimeError(f"Missing required database environment variables: {missing}")

    return psycopg2.connect(
        host=os.environ["DB_HOST"],
        port=os.environ["DB_PORT"],
        dbname=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
    )
