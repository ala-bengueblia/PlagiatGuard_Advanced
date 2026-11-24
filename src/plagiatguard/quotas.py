from datetime import datetime, timezone

from .config import DAILY_QUOTA
from .storage import connect


def consume_quota(username: str) -> tuple[bool, int]:
    day = datetime.now(timezone.utc).date().isoformat()

    with connect() as conn:
        row = conn.execute(
            "SELECT count FROM quota WHERE username=? AND day=?",
            (username, day),
        ).fetchone()

        count = row["count"] if row else 0

        if count >= DAILY_QUOTA:
            return False, 0

        conn.execute(
            """INSERT INTO quota(username, day, count)
            VALUES (?, ?, 1)
            ON CONFLICT(username, day)
            DO UPDATE SET count=count+1""",
            (username, day),
        )

        return True, DAILY_QUOTA - count - 1