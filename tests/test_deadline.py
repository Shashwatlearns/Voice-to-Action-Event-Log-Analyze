from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from voice_to_action.deadline import extract_deadline


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Rahul needs a VGA cable by 4 PM", datetime(2026, 9, 11, 16, 0)),
        ("bring it at 16:30", datetime(2026, 9, 11, 16, 30)),
        ("before 6.30 pm please", datetime(2026, 9, 11, 18, 30)),
        ("we need coffee in 20 minutes", datetime(2026, 9, 11, 14, 20)),
        ("ice for the counter in an hour", datetime(2026, 9, 11, 15, 0)),
        ("send the projector by noon", datetime(2026, 9, 11, 12, 0)),
        ("badges tomorrow 9am", datetime(2026, 9, 12, 9, 0)),
        ("flyers by end of day", datetime(2026, 9, 11, 18, 0)),
        ("chairs by eod", datetime(2026, 9, 11, 18, 0)),
        ("photographer at the entrance by 5", datetime(2026, 9, 11, 17, 0)),
        ("tonight at 8", datetime(2026, 9, 11, 20, 0)),
        ("tomorrow by 4", datetime(2026, 9, 12, 16, 0)),
        ("coffee in the lobby at 11", datetime(2026, 9, 12, 11, 0)),
    ],
)
def test_deadline_resolution(text: str, expected: datetime, ref: datetime) -> None:
    match = extract_deadline(text, ref)
    assert match is not None, text
    assert match.when == expected


def test_asap_is_near_term_and_flagged(ref: datetime) -> None:
    match = extract_deadline("2 mics needed at the desk asap", ref)
    assert match is not None
    assert match.kind == "asap"
    assert match.text == "ASAP"
    assert timedelta(0) < match.when - ref <= timedelta(minutes=30)


def test_plain_numbers_are_not_deadlines(ref: datetime) -> None:
    assert extract_deadline("Priya wants 20 chairs", ref) is None
    assert extract_deadline("send it to Room 204", ref) is None
    assert extract_deadline("escort them by 2 volunteers", ref) is None


def test_deadline_text_and_span(ref: datetime) -> None:
    text = "Rahul needs a VGA cable by 4 PM"
    match = extract_deadline(text, ref)
    assert match is not None
    assert match.text == "by 4 PM"
    assert text[match.start : match.end] == "by 4 PM"


def test_bare_tomorrow_defaults_to_end_of_day(ref: datetime) -> None:
    match = extract_deadline("print the banners tomorrow", ref, eod_hour=17)
    assert match is not None
    assert match.when == datetime(2026, 9, 12, 17, 0)
