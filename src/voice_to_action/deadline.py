"""Deterministic deadline extraction for event updates.

Handles phrases such as ``by 4 PM``, ``at 16:30``, ``before noon``, ``in 20 minutes``,
``tomorrow 9am``, ``by end of day`` and ``asap``. All datetimes are naive and relative to a
caller supplied ``reference`` time so results are reproducible in tests.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta

from voice_to_action import config

TRIGGER = r"(?P<trigger>by|at|before|until|till|around|due|no later than|latest by|not later than)"

DAY_RE = re.compile(
    r"\b(?P<day>day after tomorrow|tomorrow|tmrw|today|tonight|this (?:morning|afternoon|evening))\b",
    re.IGNORECASE,
)
RELATIVE_RE = re.compile(
    r"\b(?:in|within)\s+(?:the\s+next\s+)?(?P<amount>\d+|half an|an?|a couple of|a few|few)\s*"
    r"(?P<unit>minutes?|mins?|hours?|hrs?|h|m)\b",
    re.IGNORECASE,
)
CLOCK_RE = re.compile(
    rf"(?:\b{TRIGGER}\s+(?:the\s+)?)?\b(?P<hour>[01]?\d|2[0-3])(?:[:.](?P<minute>[0-5]\d))?"
    r"\s*(?P<ampm>a\.?m\.?|p\.?m\.?)?(?:\s*(?P<suffix>o'?clock|hrs|hours))?\b",
    re.IGNORECASE,
)
NAMED_RE = re.compile(
    rf"(?:\b{TRIGGER}\s+(?:the\s+)?)?"
    r"\b(?P<word>noon|midday|midnight|eod|end of (?:the )?day|close of play|lunchtime|lunch time)\b",
    re.IGNORECASE,
)
ASAP_RE = re.compile(
    r"\b(?P<word>asap|a\.s\.a\.p\.?|immediately|right now|right away|straight away|urgently|now)\b",
    re.IGNORECASE,
)

# Words allowed directly after a bare "by 4" style match (otherwise "by 2 volunteers" would be a time)
_ALLOWED_FOLLOWERS = frozenset({
    "today", "tonight", "tomorrow", "tmrw", "sharp", "latest", "at", "in", "to", "for", "please", "and",
    "or", "so", "because", "if", "since", "before", "this", "the", "otherwise", "max", "maximum", "then",
    "as", "on", "with", "pls", "plz", "thanks", "thank", "itself",
})
_ASAP_MINUTES = 15


@dataclass(frozen=True)
class DeadlineMatch:
    """Result of :func:`extract_deadline`."""

    when: datetime | None
    text: str
    start: int
    end: int
    kind: str  # clock | relative | named | asap
    spans: tuple[tuple[int, int], ...]

    @property
    def char_spans(self) -> list[tuple[int, int]]:
        return list(self.spans)


def _floor_minute(value: datetime) -> datetime:
    return value.replace(second=0, microsecond=0)


def _day_info(day_word: str | None) -> tuple[int, str | None]:
    """Return ``(day_offset, am_pm_hint)`` for a day word."""
    if not day_word:
        return 0, None
    word = day_word.lower()
    if word.startswith("day after"):
        return 2, None
    if word in ("tomorrow", "tmrw"):
        return 1, None
    if word == "tonight" or "afternoon" in word or "evening" in word:
        return 0, "pm"
    if "morning" in word:
        return 0, "am"
    return 0, None


def _disambiguate_hour(
    hour: int, minute: int, base: datetime, reference: datetime, has_day: bool, hint: str | None
) -> int:
    """Resolve a 1-12 hour without AM/PM into a 24h hour."""
    if hour > 12:
        return hour
    if hint == "am":
        return 0 if hour == 12 else hour
    if hint == "pm":
        return 12 if hour == 12 else hour + 12
    if has_day:
        # "tomorrow by 4" almost always means the afternoon; 7-12 stay in the morning.
        return hour if 7 <= hour <= 12 else hour + 12
    cand_am = hour % 12
    cand_pm = cand_am + 12
    for candidate in (cand_am, cand_pm):
        if base.replace(hour=candidate, minute=minute) >= reference:
            return candidate
    return cand_pm


def _relative_amount(raw: str) -> float:
    raw = raw.lower().strip()
    if raw.isdigit():
        return float(raw)
    return {"half an": 0.5, "a": 1.0, "an": 1.0, "a couple of": 2.0, "a few": 3.0, "few": 3.0}.get(raw, 1.0)


def _combine_text(
    text: str, primary: re.Match[str], day: re.Match[str] | None
) -> tuple[str, tuple[tuple[int, int], ...]]:
    spans: list[tuple[int, int]] = [(primary.start(), primary.end())]
    label = primary.group(0).strip()
    if day is not None:
        spans.append((day.start(), day.end()))
        lo, hi = min(primary.start(), day.start()), max(primary.end(), day.end())
        if hi - lo <= (primary.end() - primary.start()) + (day.end() - day.start()) + 3:
            label = text[lo:hi].strip()
        else:
            label = f"{label} {day.group(0)}"
    return label.strip(" ,.;:"), tuple(spans)


def extract_deadline(
    text: str,
    reference: datetime | None = None,
    eod_hour: int | None = None,
) -> DeadlineMatch | None:
    """Find the deadline phrase in ``text`` and resolve it relative to ``reference``."""
    reference = _floor_minute(reference or datetime.now())
    eod_hour = config.END_OF_DAY_HOUR if eod_hour is None else eod_hour

    day_match = DAY_RE.search(text)
    day_offset, hint = _day_info(day_match.group("day") if day_match else None)
    base = reference + timedelta(days=day_offset)

    # 1. relative: "in 20 minutes", "within an hour"
    rel = RELATIVE_RE.search(text)
    if rel:
        amount = _relative_amount(rel.group("amount"))
        unit = rel.group("unit").lower()
        delta = timedelta(hours=amount) if unit.startswith("h") else timedelta(minutes=amount)
        when = _floor_minute(reference + delta)
        span = (rel.start(), rel.end())
        return DeadlineMatch(when, rel.group(0).strip(), rel.start(), rel.end(), "relative", (span,))

    # 2. clock time: "by 4 PM", "at 16:30", "before 6.30", "by 4"
    for m in CLOCK_RE.finditer(text):
        explicit = bool(m.group("ampm") or m.group("minute") or m.group("suffix"))
        if not explicit and not m.group("trigger"):
            continue
        if not explicit:
            following = text[m.end():].lstrip()
            if following and following[0].isalpha():
                next_word = re.split(r"\W+", following, maxsplit=1)[0].lower()
                if next_word not in _ALLOWED_FOLLOWERS:
                    continue
        hour = int(m.group("hour"))
        minute = int(m.group("minute") or 0)
        ampm = m.group("ampm")
        if ampm:
            ampm = ampm.lower().replace(".", "")
            hour = hour % 12
            if ampm == "pm":
                hour += 12
        else:
            hour = _disambiguate_hour(hour, minute, base, reference, day_match is not None, hint)
        when = base.replace(hour=hour % 24, minute=minute)
        label, spans = _combine_text(text, m, day_match)
        return DeadlineMatch(when, label, m.start(), m.end(), "clock", spans)

    # 3. named times: noon, midnight, end of day
    named = NAMED_RE.search(text)
    if named:
        word = named.group("word").lower()
        if word in ("noon", "midday"):
            when = base.replace(hour=12, minute=0)
        elif word == "midnight":
            when = (base + timedelta(days=1)).replace(hour=0, minute=0)
        elif word.startswith("lunch"):
            when = base.replace(hour=13, minute=0)
        else:  # eod / end of day / close of play
            when = base.replace(hour=eod_hour, minute=0)
        label, spans = _combine_text(text, named, day_match)
        return DeadlineMatch(when, label, named.start(), named.end(), "named", spans)

    # 4. ASAP-style: treated as due in a few minutes so it sorts to the top
    asap = ASAP_RE.search(text)
    if asap:
        when = reference + timedelta(minutes=_ASAP_MINUTES)
        return DeadlineMatch(when, "ASAP", asap.start(), asap.end(), "asap", ((asap.start(), asap.end()),))

    # 5. a bare day word ("tomorrow") without a time -> default to end of that day
    if day_match and day_offset > 0:
        when = base.replace(hour=eod_hour, minute=0)
        span = (day_match.start(), day_match.end())
        return DeadlineMatch(when, day_match.group(0), day_match.start(), day_match.end(), "named", (span,))
    return None
