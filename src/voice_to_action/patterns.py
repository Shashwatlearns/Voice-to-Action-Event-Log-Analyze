"""Domain vocabulary for live-event logistics.

These patterns feed a spaCy ``EntityRuler`` so that domain terms ("VGA cable", "Hall B")
are recognised deterministically, complementing the statistical NER model.
"""

from __future__ import annotations

import re
from typing import Any

# --------------------------------------------------------------------------- items
ITEM_PHRASES: tuple[str, ...] = (
    # AV / tech
    "cable", "vga cable", "hdmi cable", "usb cable", "ethernet cable", "lan cable", "aux cable",
    "xlr cable", "charging cable", "vga", "hdmi", "adapter", "hdmi adapter", "vga adapter", "dongle",
    "projector", "screen", "projector screen", "monitor", "display", "tv", "laptop", "tablet",
    "microphone", "mic", "wireless mic", "lapel mic", "handheld mic", "headset", "speaker",
    "mixer", "amplifier", "clicker", "presenter remote", "remote", "camera", "tripod", "light",
    "spotlight", "stage light", "walkie talkie", "radio", "wifi", "wifi password", "internet",
    "printer", "scanner",
    # furniture / staging
    "chair", "table", "podium", "lectern", "stand", "mic stand", "whiteboard", "flipchart",
    "easel", "sofa", "stool", "bench", "barricade", "rope", "carpet", "backdrop", "stage riser",
    # stationery / printing / signage
    "marker", "pen", "notepad", "paper", "tape", "duct tape", "scissors", "stapler", "banner",
    "poster", "sign", "signage", "badge", "lanyard", "name tag", "printout", "flyer", "brochure",
    "programme", "program", "schedule", "certificate", "trophy", "gift", "bouquet", "flowers",
    # catering
    "water", "water bottle", "bottle", "coffee", "tea", "juice", "snack", "lunch", "dinner",
    "breakfast", "sandwich", "ice", "cup", "plate", "cutlery", "napkin", "food",
    # power / electrical
    "extension cord", "extension cable", "power strip", "power bank", "charger", "battery",
    "generator", "plug", "socket",
    # personnel
    "volunteer", "security guard", "security", "photographer", "videographer", "technician",
    "electrician", "usher", "host", "translator", "interpreter", "medic", "driver", "staff",
    "helper",
    # transport
    "cab", "taxi", "car", "van", "shuttle", "bus", "trolley", "cart", "ladder",
    # safety / misc
    "first aid kit", "fire extinguisher", "sanitizer", "tissue", "mask", "umbrella", "key",
    "garbage bag", "bin", "dustbin", "broom", "mop",
)

# ----------------------------------------------------------------------- locations
LOCATION_PHRASES: tuple[str, ...] = (
    "main stage", "stage", "backstage", "registration desk", "registration", "reception",
    "help desk", "info desk", "green room", "vip lounge", "lounge", "lobby", "foyer", "entrance",
    "main entrance", "exit", "parking lot", "parking", "auditorium", "food court", "dining area",
    "dining hall", "drinks counter", "cafeteria", "kitchen", "pantry", "washroom", "restroom",
    "control room", "av room", "sound booth", "media room", "press room", "exhibition hall",
    "expo hall", "conference room", "boardroom", "seminar hall", "workshop room", "terrace",
    "rooftop", "lawn", "garden", "ground floor", "first floor", "second floor", "basement",
    "back office", "storage room", "store room", "warehouse", "loading dock", "gate",
)

# Words that can be followed by an identifier: "Hall B", "Room 204", "Booth 12", "Gate 3"
NUMBERED_LOCATION_WORDS: tuple[str, ...] = (
    "hall", "room", "booth", "gate", "stage", "tent", "zone", "block", "table", "counter", "floor", "wing", "area",
)

# --------------------------------------------------------------------- categories
# Ordered: the first category with a keyword hit wins. Keywords are single lowercase words
# matched against (crudely singularised) words of the item / sentence.
CATEGORY_KEYWORDS: dict[str, frozenset[str]] = {
    "av_equipment": frozenset({
        "cable", "vga", "hdmi", "usb", "aux", "xlr", "adapter", "dongle", "projector", "screen",
        "monitor", "display", "tv", "laptop", "tablet", "microphone", "mic", "headset", "speaker",
        "mixer", "amplifier", "clicker", "remote", "camera", "tripod", "light", "spotlight",
        "walkie", "radio", "wifi", "internet", "printer", "scanner", "audio", "video", "sound",
    }),
    "power_electrical": frozenset({
        "extension", "cord", "strip", "powerbank", "charger", "battery", "batterie", "generator",
        "plug", "socket", "power", "electricity",
    }),
    "furniture": frozenset({
        "chair", "table", "podium", "lectern", "stand", "whiteboard", "flipchart", "easel", "sofa",
        "stool", "bench", "barricade", "rope", "carpet", "backdrop", "riser",
    }),
    "catering": frozenset({
        "water", "bottle", "coffee", "tea", "juice", "snack", "lunch", "dinner", "breakfast",
        "sandwich", "ice", "cup", "plate", "cutlery", "napkin", "food", "drink", "catering", "meal",
    }),
    "signage_printing": frozenset({
        "banner", "poster", "sign", "signage", "badge", "lanyard", "tag", "printout", "flyer",
        "brochure", "programme", "program", "schedule", "certificate", "print", "marker", "pen",
        "notepad", "paper", "tape", "scissor", "stapler",
    }),
    "personnel": frozenset({
        "volunteer", "guard", "security", "photographer", "videographer", "technician",
        "electrician", "usher", "host", "translator", "interpreter", "medic", "driver", "staff",
        "helper", "people", "person", "crew",
    }),
    "transport": frozenset({"cab", "taxi", "car", "van", "shuttle", "bus", "trolley", "cart", "pickup", "ride"}),
    "safety_medical": frozenset({"aid", "kit", "extinguisher", "sanitizer", "mask", "medical", "ambulance"}),
    "hospitality": frozenset({"trophy", "trophie", "gift", "bouquet", "flower", "umbrella", "key", "welcome"}),
    "housekeeping": frozenset({"garbage", "bin", "dustbin", "broom", "mop", "tissue", "clean", "cleaning"}),
}

# ------------------------------------------------------------------ action verbs
ACTION_VERBS: frozenset[str] = frozenset({
    "need", "want", "require", "request", "ask", "bring", "send", "deliver", "get", "fetch", "grab",
    "arrange", "set", "setup", "move", "shift", "order", "book", "call", "fix", "repair", "replace",
    "check", "print", "prepare", "collect", "pick", "drop", "install", "remove", "clean", "refill",
    "restock", "provide", "supply", "place", "put", "carry", "take", "find", "organise", "organize",
    "connect", "test", "charge", "escort", "assign", "buy", "purchase", "hand",
})

# ------------------------------------------------------------------- misc lexicon
NUMBER_WORDS: dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "dozen": 12, "couple": 2, "pair": 2,
}

STRIP_LEADING_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "some", "more", "extra", "another", "few", "couple", "of", "additional",
    "new", "spare", "that", "this", "those", "these", "his", "her", "their", "our", "my", "any",
})

# Capitalised words that are never a person's name
PERSON_STOPWORDS: frozenset[str] = frozenset({
    "team", "someone", "somebody", "anyone", "everyone", "please", "urgent", "asap", "note",
    "update", "reminder", "fyi", "we", "i", "they", "you", "registration", "backstage", "stage",
    "hall", "room", "booth", "gate", "catering", "security", "volunteers", "volunteer", "staff",
    "sir", "madam", "ma'am", "guest", "guests", "speaker", "speakers", "keynote", "today",
    "tomorrow", "tonight", "morning", "afternoon", "evening", "chief", "vip", "the", "a", "an",
})

HIGH_PRIORITY_RE = re.compile(
    r"\b(urgent(?:ly)?|asap|a\.s\.a\.p\.?|immediately|critical|emergency|right now|right away|straight away|"
    r"top priority|high priority|blocker|blocking)\b",
    re.IGNORECASE,
)
LOW_PRIORITY_RE = re.compile(
    r"\b(whenever|no rush|no hurry|low priority|"
    r"when (?:you|she|he|they) (?:get|gets|have|has) (?:a|the) (?:chance|time)|"
    r"when free|eventually|sometime|not urgent|nice to have|if possible)\b",
    re.IGNORECASE,
)
DONE_RE = re.compile(
    r"\b(?:is|are|was|were|has been|have been|got|already|been|now)\s+"
    r"(?:done|delivered|completed|sorted|handled|resolved|finished|arranged|fixed|set up|taken care of|in place)\b"
    r"|\b(?:done|delivered|completed|sorted|resolved|finished|fixed)\s*[.!]*\s*$"
    r"|^\s*(?:done|completed|resolved|delivered)\b",
    re.IGNORECASE,
)
IN_PROGRESS_RE = re.compile(
    r"\b(in progress|working on (?:it|the|this)|on it|on the way|on its way|being (?:delivered|set up|arranged)|wip)\b",
    re.IGNORECASE,
)


# ------------------------------------------------------------------ pattern build
def _token_patterns(phrase: str) -> list[dict[str, Any]]:
    """Turn ``"vga cable"`` into case-insensitive token patterns with plural variants."""
    words = phrase.lower().split()
    pattern: list[dict[str, Any]] = [{"LOWER": w} for w in words[:-1]]
    last = words[-1]
    variants = {last, last + "s", last + "es"}
    if last.endswith("y") and len(last) > 2:
        variants.add(last[:-1] + "ies")
    pattern.append({"LOWER": {"IN": sorted(variants)}})
    return pattern


def build_entity_patterns() -> list[dict[str, Any]]:
    """Patterns for the spaCy ``EntityRuler`` (labels: ``ITEM`` and ``LOCATION``)."""
    patterns: list[dict[str, Any]] = []
    for phrase in ITEM_PHRASES:
        patterns.append({"label": "ITEM", "pattern": _token_patterns(phrase)})
    for phrase in LOCATION_PHRASES:
        patterns.append({"label": "LOCATION", "pattern": _token_patterns(phrase)})
    identifier = {"TEXT": {"REGEX": r"^(?:[A-Z]|\d{1,3}[A-Za-z]?)$"}}
    for word in NUMBERED_LOCATION_WORDS:
        patterns.append({"label": "LOCATION", "pattern": [{"LOWER": word}, identifier]})
        numbered = [{"LOWER": word}, {"LOWER": "no"}, {"IS_PUNCT": True, "OP": "?"}, identifier]
        patterns.append({"label": "LOCATION", "pattern": numbered})
    return patterns


def singularize(word: str) -> str:
    """Very small heuristic singulariser sufficient for category matching."""
    w = word.lower().strip(".,;:!?()[]'\"")
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("ches") or w.endswith("shes") or w.endswith("xes"):
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


def categorize(*texts: str | None) -> str:
    """Return the first matching category for the given texts (checked in order)."""
    for text in texts:
        if not text:
            continue
        words = {singularize(w) for w in text.split()}
        words |= {w.lower().strip(".,;:!?") for w in text.split()}
        for category, keywords in CATEGORY_KEYWORDS.items():
            if words & keywords:
                return category
    return "other"
