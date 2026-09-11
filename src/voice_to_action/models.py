"""Data model for a parsed event task."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

PRIORITIES: tuple[str, ...] = ("low", "medium", "high")
STATUSES: tuple[str, ...] = ("pending", "in_progress", "done")
DUE_STATES: tuple[str, ...] = ("overdue", "due_soon", "on_track", "done", "none")

CSV_FIELDS: tuple[str, ...] = (
    "id",
    "person",
    "item",
    "quantity",
    "location",
    "deadline",
    "deadline_text",
    "action",
    "priority",
    "category",
    "status",
    "confidence",
    "raw_text",
    "created_at",
)


def _new_id() -> str:
    return uuid4().hex[:8]


def _parse_dt(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


@dataclass
class Task:
    """A single actionable logistics task extracted from an event update."""

    raw_text: str
    person: str | None = None
    item: str | None = None
    quantity: int | None = None
    location: str | None = None
    deadline: datetime | None = None
    deadline_text: str | None = None
    action: str | None = None
    priority: str = "medium"
    category: str = "other"
    status: str = "pending"
    confidence: float = 0.0
    id: str = field(default_factory=_new_id)
    created_at: datetime = field(default_factory=datetime.now)

    # ------------------------------------------------------------------ helpers
    def is_overdue(self, now: datetime | None = None) -> bool:
        if self.deadline is None or self.status == "done":
            return False
        return self.deadline < (now or datetime.now())

    def due_state(self, now: datetime | None = None, soon_minutes: int = 15) -> str:
        """Return one of :data:`DUE_STATES` describing how urgent the task is right now."""
        if self.status == "done":
            return "done"
        if self.deadline is None:
            return "none"
        now = now or datetime.now()
        if self.deadline < now:
            return "overdue"
        if self.deadline - now <= timedelta(minutes=soon_minutes):
            return "due_soon"
        return "on_track"

    def summary(self) -> str:
        """Human readable one-liner, e.g. ``Rahul -> 1x VGA cable @ Hall B by 16:00``."""
        who = self.person or "(unassigned)"
        what = self.item or "(unknown item)"
        if self.quantity:
            what = f"{self.quantity}x {what}"
        where = f" @ {self.location}" if self.location else ""
        when = f" by {self.deadline:%H:%M}" if self.deadline else ""
        return f"{who} -> {what}{where}{when} [{self.priority}]"

    # ------------------------------------------------------------ serialisation
    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["deadline"] = self.deadline.isoformat() if self.deadline else None
        data["created_at"] = self.created_at.isoformat()
        return data

    def to_row(self) -> dict[str, Any]:
        """Flat dict in :data:`CSV_FIELDS` order (used for CSV export)."""
        data = self.to_dict()
        return {key: data.get(key) for key in CSV_FIELDS}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Task:
        known = {f for f in cls.__dataclass_fields__}
        payload = {k: v for k, v in data.items() if k in known}
        payload["deadline"] = _parse_dt(payload.get("deadline"))
        payload["created_at"] = _parse_dt(payload.get("created_at")) or datetime.now()
        if payload.get("quantity") is not None:
            payload["quantity"] = int(payload["quantity"])
        payload["confidence"] = float(payload.get("confidence") or 0.0)
        return cls(**payload)
