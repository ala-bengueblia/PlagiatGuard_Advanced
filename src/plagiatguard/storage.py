import hashlib
import json
import sqlite3
from datetime import datetime, timezone

from .config import DATABASE_PATH


def connect():
    conn = sqlite3.connect(DATABASE_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with connect() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS users(
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'analyst',
            active INTEGER NOT NULL DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS analyses(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            filename TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            score REAL NOT NULL,
            report_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS audit(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            action TEXT NOT NULL,
            created_at TEXT NOT NULL,
            details TEXT
        );

        CREATE TABLE IF NOT EXISTS quota(
            username TEXT NOT NULL,
            day TEXT NOT NULL,
            count INTEGER NOT NULL,
            PRIMARY KEY(username, day)
        );
        """)


def hash_content(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def audit(username, action, details=""):
    with connect() as conn:
        conn.execute(
            """INSERT INTO audit
            (username, action, created_at, details)
            VALUES (?, ?, ?, ?)""",
            (
                username,
                action,
                datetime.now(timezone.utc).isoformat(),
                details[:1000],
            ),
        )


def save_analysis(username, filename, content, score, report):
    with connect() as conn:
        conn.execute(
            """INSERT INTO analyses
            (username, filename, content_hash, created_at, score, report_json)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (
                username,
                filename,
                hash_content(content),
                datetime.now(timezone.utc).isoformat(),
                score,
                json.dumps(report, ensure_ascii=False),
            ),
        )