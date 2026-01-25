import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

import sql_pipeline.sql_retrieval as sql_retrieval  # noqa: WPS433


def create_events_table() -> None:
    """
    Create a table for weekly events in the existing MySQL database.

    Uses the same connection settings as the sql_pipeline pipeline
    (MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB).
    """
    ddl = """
    DROP TABLE IF EXISTS weekly_events;
    CREATE TABLE weekly_events (
        id INT AUTO_INCREMENT PRIMARY KEY,
        source_pdf VARCHAR(255) NULL,
        page_number INT NULL,
        event_name VARCHAR(255) NOT NULL,
        event_date VARCHAR(255) NOT NULL,
        start_date DATE NULL,
        end_date DATE NULL,
        start_time TIME NULL,
        end_time TIME NULL,
        raw_text TEXT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """

    conn = sql_retrieval._get_db_connection()
    try:
        with conn.cursor() as cur:
            # Drop and recreate table each time this script runs
            for stmt in ddl.split(";"):
                if stmt.strip():
                    cur.execute(stmt)
        print("weekly_events table dropped and recreated.")
    finally:
        conn.close()


def main() -> None:
    try:
        create_events_table()
    except Exception as exc:  # noqa: BLE001
        print(f"Error creating weekly_events table: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
