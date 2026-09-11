"""SQLite persistence for tasks plus CSV / JSON export."""

from __future__ import annotations

import csv
import io
import json
import sqlite3
import threading
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from voice_to_action import config
from voice_to_action.models import CSV_FIELDS, STATUSES, Task

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id            TEXT PRIMARY KEY,
    raw_text      TEXT NOT NULL,
    person        TEXT,
    item          TEXT,
    quantity      INTEGER,
    location      TEXT,
    deadline      TEXT,
    deadline_text TEXT,
    action        TEXT,
    priority      TEXT NOT NULL DEFAULT 'medium',
    category      TEXT NOT NULL DEFAULT 'other',
    status        TEXT NOT NULL DEFAULT 'pending',
    confidence    REAL NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_deadline ON tasks(deadline);
"""

_COLUMNS = (
    "id", "raw_text", "person", "item", "quantity", "location", "deadline", "deadline_text",
    "action", "priority", "category", "status", "confidence", "created_at",
)
_ORDER = "ORDER BY CASE WHEN deadline IS NULL THEN 1 ELSE 0 END, deadline, created_at"


class TaskStore:
    """Thread-safe SQLite backed task repository. Use ``":memory:"`` for an ephemeral store."""

    def __init__(self, path: str | Path = config.DB_PATH) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock, self._conn:
            self._conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------ writes
    def add(self, task: Task) -> Task:
        self.add_many([task])
        return task

    def add_many(self, tasks: Iterable[Task]) -> int:
        rows = [tuple(task.to_dict()[col] for col in _COLUMNS) for task in tasks]
        if not rows:
            return 0
        placeholders = ", ".join("?" for _ in _COLUMNS)
        sql = f"INSERT OR REPLACE INTO tasks ({', '.join(_COLUMNS)}) VALUES ({placeholders})"
        with self._lock, self._conn:
            self._conn.executemany(sql, rows)
        return len(rows)

    def update(self, task: Task) -> None:
        self.add(task)

    def update_fields(self, task_id: str, **fields: Any) -> bool:
        allowed = {k: v for k, v in fields.items() if k in _COLUMNS and k != "id"}
        if not allowed:
            return False
        if "deadline" in allowed and isinstance(allowed["deadline"], datetime):
            allowed["deadline"] = allowed["deadline"].isoformat()
        if "status" in allowed and allowed["status"] not in STATUSES:
            raise ValueError(f"Invalid status {allowed['status']!r}; expected one of {STATUSES}")
        assignments = ", ".join(f"{k} = ?" for k in allowed)
        with self._lock, self._conn:
            cur = self._conn.execute(f"UPDATE tasks SET {assignments} WHERE id = ?", (*allowed.values(), task_id))
        return cur.rowcount > 0

    def update_status(self, task_id: str, status: str) -> bool:
        return self.update_fields(task_id, status=status)

    def delete(self, task_id: str) -> bool:
        with self._lock, self._conn:
            cur = self._conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        return cur.rowcount > 0

    def clear_done(self) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute("DELETE FROM tasks WHERE status = 'done'")
        return cur.rowcount

    def clear_all(self) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute("DELETE FROM tasks")
        return cur.rowcount

    # ------------------------------------------------------------------- reads
    def get(self, task_id: str) -> Task | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        return Task.from_dict(dict(row)) if row else None

    def list_tasks(
        self,
        status: str | Iterable[str] | None = None,
        priority: str | Iterable[str] | None = None,
        category: str | Iterable[str] | None = None,
        person: str | None = None,
    ) -> list[Task]:
        clauses: list[str] = []
        params: list[Any] = []
        for column, value in (("status", status), ("priority", priority), ("category", category)):
            if value is None:
                continue
            values = [value] if isinstance(value, str) else list(value)
            if not values:
                continue
            clauses.append(f"{column} IN ({', '.join('?' for _ in values)})")
            params.extend(values)
        if person:
            clauses.append("LOWER(person) LIKE ?")
            params.append(f"%{person.lower()}%")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._conn.execute(f"SELECT * FROM tasks {where} {_ORDER}", params).fetchall()
        return [Task.from_dict(dict(row)) for row in rows]

    def count(self, status: str | None = None) -> int:
        sql = "SELECT COUNT(*) FROM tasks" + (" WHERE status = ?" if status else "")
        with self._lock:
            row = self._conn.execute(sql, (status,) if status else ()).fetchone()
        return int(row[0])

    # ----------------------------------------------------------------- exports
    def export_csv(self, path: str | Path | None = None, tasks: Iterable[Task] | None = None) -> str:
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        for task in tasks if tasks is not None else self.list_tasks():
            writer.writerow(task.to_row())
        content = buffer.getvalue()
        if path:
            Path(path).write_text(content, encoding="utf-8")
        return content

    def export_json(self, path: str | Path | None = None, tasks: Iterable[Task] | None = None) -> str:
        payload = [task.to_dict() for task in (tasks if tasks is not None else self.list_tasks())]
        content = json.dumps(payload, indent=2, ensure_ascii=False)
        if path:
            Path(path).write_text(content, encoding="utf-8")
        return content

    # --------------------------------------------------------------- lifecycle
    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> TaskStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
