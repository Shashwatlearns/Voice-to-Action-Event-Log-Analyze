from __future__ import annotations

from datetime import datetime

import pytest

from voice_to_action.parser import EventLogParser, normalize_text


def test_basic_request(parser: EventLogParser, ref: datetime) -> None:
    task = parser.parse("Rahul needs a VGA cable by 4 PM", reference_time=ref)
    assert task.person == "Rahul"
    assert task.item is not None and task.item.lower() == "vga cable"
    assert task.deadline == datetime(2026, 9, 11, 16, 0)
    assert task.deadline_text == "by 4 PM"
    assert task.action == "need"
    assert task.category == "av_equipment"
    assert task.status == "pending"
    assert task.confidence >= 0.8


def test_quantity_and_location(parser: EventLogParser, ref: datetime) -> None:
    task = parser.parse("Priya wants 20 chairs in Hall B before 6:30", reference_time=ref)
    assert task.person == "Priya"
    assert task.quantity == 20
    assert task.item is not None and "chair" in task.item.lower()
    assert task.location == "Hall B"
    assert task.deadline == datetime(2026, 9, 11, 18, 30)
    assert task.category == "furniture"


def test_imperative_without_person(parser: EventLogParser, ref: datetime) -> None:
    task = parser.parse("Send the projector to the main stage by noon", reference_time=ref)
    assert task.person is None
    assert task.item is not None and task.item.lower() == "projector"
    assert task.location is not None and task.location.lower() == "main stage"
    assert task.deadline == datetime(2026, 9, 11, 12, 0)
    assert task.action == "send"


def test_urgent_request_is_high_priority(parser: EventLogParser, ref: datetime) -> None:
    task = parser.parse("URGENT: 2 wireless mics needed at the registration desk asap", reference_time=ref)
    assert task.priority == "high"
    assert task.quantity == 2
    assert task.item is not None and "mic" in task.item.lower()
    assert task.location is not None and task.location.lower() == "registration desk"
    assert task.category == "av_equipment"


def test_low_priority_and_requested_person(parser: EventLogParser, ref: datetime) -> None:
    task = parser.parse("Ask Meera to bring water bottles whenever she gets a chance", reference_time=ref)
    assert task.person == "Meera"
    assert task.item is not None and task.item.lower() == "water bottles"
    assert task.priority == "low"
    assert task.category == "catering"


def test_word_quantities_and_numbered_location(parser: EventLogParser, ref: datetime) -> None:
    task = parser.parse("Kabir needs two volunteers at Gate 3 right now", reference_time=ref)
    assert task.quantity == 2
    assert task.location == "Gate 3"
    assert task.priority == "high"
    assert task.category == "personnel"


def test_relative_deadline_and_power_category(parser: EventLogParser, ref: datetime) -> None:
    task = parser.parse("Arjun needs 3 extension cords in Room 204 within 30 minutes", reference_time=ref)
    assert task.quantity == 3
    assert task.item is not None and "extension cord" in task.item.lower()
    assert task.location == "Room 204"
    assert task.deadline == datetime(2026, 9, 11, 14, 30)
    assert task.category == "power_electrical"
    assert task.priority == "high"  # due within 30 minutes


def test_done_status_detection(parser: EventLogParser, ref: datetime) -> None:
    task = parser.parse("The projector in Hall A is delivered", reference_time=ref)
    assert task.status == "done"
    assert task.location == "Hall A"


def test_in_progress_status(parser: EventLogParser, ref: datetime) -> None:
    task = parser.parse("Working on the banner for the entrance, done by 6", reference_time=ref)
    assert task.status in {"in_progress", "done"}
    assert task.deadline == datetime(2026, 9, 11, 18, 0)


def test_beneficiary_is_not_the_item(parser: EventLogParser, ref: datetime) -> None:
    task = parser.parse("Bring a HDMI cable for the projector by 3pm", reference_time=ref)
    assert task.item is not None and task.item.lower().startswith("hdmi cable")


def test_parse_many_skips_blank_lines(parser: EventLogParser, ref: datetime) -> None:
    text = "\nRahul needs a VGA cable by 4 PM\n\n- Send the projector to the main stage by noon\n   \n"
    tasks = parser.parse_many(text, reference_time=ref)
    assert len(tasks) == 2
    assert tasks[0].raw_text == "Rahul needs a VGA cable by 4 PM"
    assert tasks[1].raw_text.startswith("Send the projector")


def test_empty_update_rejected(parser: EventLogParser) -> None:
    with pytest.raises(ValueError):
        parser.parse("   ")


def test_explain_returns_html(parser: EventLogParser) -> None:
    html = parser.explain("Rahul needs a VGA cable by 4 PM")
    assert "<div" in html
    assert "ITEM" in html


def test_normalize_text() -> None:
    assert normalize_text("  Rahul   needs\ta cable \n") == "Rahul needs a cable"
