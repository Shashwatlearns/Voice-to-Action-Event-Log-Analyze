from __future__ import annotations

import csv
import io
import json
from datetime import datetime

import pytest

from voice_to_action.models import Task
from voice_to_action.storage import TaskStore


@pytest.fixture
def store() -> TaskStore:
    with TaskStore(":memory:") as s:
        yield s


def _task(**overrides: object) -> Task:
    base = {
        "raw_text": "Rahul needs a VGA cable by 4 PM",
        "person": "Rahul",
        "item": "VGA cable",
        "deadline": datetime(2026, 9, 11, 16, 0),
        "deadline_text": "by 4 PM",
        "category": "av_equipment",
        "priority": "medium",
    }
    base.update(overrides)
    return Task(**base)  # type: ignore[arg-type]


def test_add_and_get_roundtrip(store: TaskStore) -> None:
    task = store.add(_task(quantity=2, confidence=0.9))
    loaded = store.get(task.id)
    assert loaded is not None
    assert loaded.person == "Rahul"
    assert loaded.item == "VGA cable"
    assert loaded.quantity == 2
    assert loaded.deadline == datetime(2026, 9, 11, 16, 0)
    assert loaded.confidence == pytest.approx(0.9)


def test_list_orders_by_deadline_with_nulls_last(store: TaskStore) -> None:
    store.add(_task(item="no deadline", deadline=None))
    store.add(_task(item="late", deadline=datetime(2026, 9, 11, 18, 0)))
    store.add(_task(item="early", deadline=datetime(2026, 9, 11, 9, 0)))
    items = [t.item for t in store.list_tasks()]
    assert items == ["early", "late", "no deadline"]


def test_filters(store: TaskStore) -> None:
    store.add(_task(person="Rahul", priority="high"))
    store.add(_task(person="Priya", priority="low", category="furniture"))
    assert len(store.list_tasks(priority="high")) == 1
    assert len(store.list_tasks(category=["furniture"])) == 1
    assert [t.person for t in store.list_tasks(person="pri")] == ["Priya"]


def test_update_status_and_clear_done(store: TaskStore) -> None:
    task = store.add(_task())
    assert store.update_status(task.id, "done") is True
    assert store.get(task.id).status == "done"  # type: ignore[union-attr]
    assert store.count(status="done") == 1
    assert store.clear_done() == 1
    assert store.count() == 0


def test_invalid_status_rejected(store: TaskStore) -> None:
    task = store.add(_task())
    with pytest.raises(ValueError):
        store.update_status(task.id, "bogus")


def test_update_fields_and_delete(store: TaskStore) -> None:
    task = store.add(_task())
    assert store.update_fields(task.id, location="Hall B", quantity=4, deadline=datetime(2026, 9, 11, 17, 0))
    loaded = store.get(task.id)
    assert loaded is not None
    assert loaded.location == "Hall B"
    assert loaded.quantity == 4
    assert loaded.deadline == datetime(2026, 9, 11, 17, 0)
    assert store.delete(task.id) is True
    assert store.get(task.id) is None


def test_export_csv_and_json(store: TaskStore, tmp_path) -> None:  # type: ignore[no-untyped-def]
    store.add(_task())
    store.add(_task(person="Priya", item="chairs", quantity=20))

    csv_text = store.export_csv(tmp_path / "tasks.csv")
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    assert len(rows) == 2
    assert {"person", "item", "deadline", "priority", "status"} <= set(rows[0])
    assert (tmp_path / "tasks.csv").exists()
    assert any(r["item"] == "chairs" and r["quantity"] == "20" for r in rows)

    payload = json.loads(store.export_json())
    assert len(payload) == 2
    assert payload[0]["deadline"] == "2026-09-11T16:00:00"
