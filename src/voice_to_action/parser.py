"""Core NLP parsing: turns free-text event updates into structured :class:`Task` objects.

Pipeline: normalise text -> deterministic deadline extraction -> spaCy (EntityRuler + NER +
dependency parse) -> heuristics for person, item, quantity, location, action, priority,
category and status.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from datetime import datetime, timedelta

import spacy
from spacy import displacy
from spacy.language import Language
from spacy.tokens import Doc, Span, Token

from voice_to_action import config
from voice_to_action.deadline import DeadlineMatch, extract_deadline
from voice_to_action.models import Task
from voice_to_action.patterns import (
    ACTION_VERBS,
    DONE_RE,
    HIGH_PRIORITY_RE,
    IN_PROGRESS_RE,
    LOW_PRIORITY_RE,
    NUMBER_WORDS,
    PERSON_STOPWORDS,
    STRIP_LEADING_WORDS,
    build_entity_patterns,
    categorize,
)

logger = logging.getLogger(__name__)

CharSpans = list[tuple[int, int]]

SPACY_LOCATION_LABELS = frozenset({"FAC", "GPE", "LOC"})
NON_ITEM_LABELS = frozenset({"PERSON", "TIME", "DATE", "LOCATION", "FAC", "GPE", "LOC", "CARDINAL", "ORDINAL"})
LOCATION_PREPOSITIONS = frozenset({"in", "at", "to", "near", "inside", "outside", "behind", "beside", "on", "from", "towards"})
PERSON_PREPOSITIONS = frozenset({"for", "from", "with", "to"})
PERSON_REQUEST_VERBS = frozenset({"ask", "tell", "remind", "inform", "notify", "call", "ping", "message"})
BENEFICIARY_PREPOSITIONS = frozenset({"for", "to"})
CUT_POS = frozenset({"ADP", "PUNCT", "SCONJ", "VERB", "AUX"})

DISPLACY_COLORS = {
    "ITEM": "#ffd166",
    "LOCATION": "#06d6a0",
    "PERSON": "#8ecae6",
    "TIME": "#f4a3b3",
    "DATE": "#f4a3b3",
    "CARDINAL": "#cdb4db",
}

_WS_RE = re.compile(r"\s+")


class ModelNotFoundError(RuntimeError):
    """Raised when the configured spaCy model is not installed."""


def load_nlp(model_name: str = config.SPACY_MODEL) -> Language:
    """Load a spaCy pipeline, raising a helpful error when the model is missing."""
    try:
        return spacy.load(model_name)
    except OSError as exc:  # model not downloaded
        raise ModelNotFoundError(
            f"spaCy model '{model_name}' is not installed. Run: python -m spacy download {model_name}"
        ) from exc


def normalize_text(text: str) -> str:
    text = text.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    return _WS_RE.sub(" ", text).strip()


def _overlaps(start: int, end: int, spans: CharSpans) -> bool:
    return any(start < s_end and end > s_start for s_start, s_end in spans)


def _token_blocked(tok: Token, spans: CharSpans) -> bool:
    return _overlaps(tok.idx, tok.idx + len(tok.text), spans)


def _span_blocked(span: Span, spans: CharSpans) -> bool:
    return _overlaps(span.start_char, span.end_char, spans)


def _clean_item(text: str) -> str | None:
    words = text.split()
    while words and (words[0].lower().strip(",.") in STRIP_LEADING_WORDS or words[0].lower() in NUMBER_WORDS or words[0].isdigit()):
        words.pop(0)
    cleaned = " ".join(words).strip(" ,.;:!?-")
    return cleaned or None


def _to_int(tok: Token) -> int | None:
    if tok.lower_ in NUMBER_WORDS:
        return NUMBER_WORDS[tok.lower_]
    if tok.like_num:
        try:
            return int(tok.text.replace(",", ""))
        except ValueError:
            return None
    return None


class EventLogParser:
    """Parse event updates into :class:`~voice_to_action.models.Task` objects.

    >>> parser = EventLogParser()
    >>> task = parser.parse("Rahul needs a VGA cable by 4 PM")
    >>> task.person, task.item
    ('Rahul', 'VGA cable')
    """

    def __init__(self, model_name: str = config.SPACY_MODEL, nlp: Language | None = None) -> None:
        self.nlp = nlp if nlp is not None else load_nlp(model_name)
        if "event_ruler" not in self.nlp.pipe_names:
            position = {"before": "ner"} if "ner" in self.nlp.pipe_names else {"last": True}
            ruler = self.nlp.add_pipe("entity_ruler", name="event_ruler", config={"overwrite_ents": True}, **position)
            ruler.add_patterns(build_entity_patterns())  # type: ignore[attr-defined]

    # ------------------------------------------------------------------ public
    def parse(self, text: str, reference_time: datetime | None = None) -> Task:
        """Parse a single update. ``reference_time`` anchors relative deadlines (defaults to now)."""
        reference = reference_time or datetime.now()
        clean = normalize_text(text)
        if not clean:
            raise ValueError("Cannot parse an empty update")

        deadline = extract_deadline(clean, reference)
        blocked: CharSpans = deadline.char_spans if deadline else []
        doc = self.nlp(clean)

        location, location_tokens = self._extract_location(doc)
        person, person_tokens = self._extract_person(doc, exclude=location_tokens)
        excluded_tokens = location_tokens | person_tokens
        item, quantity = self._extract_item(doc, blocked, excluded_tokens)
        action = self._extract_action(doc)
        status = self._detect_status(clean)
        priority = self._detect_priority(clean, deadline, reference)
        category = categorize(item, clean)
        confidence = self._confidence(person, item, deadline, location)

        return Task(
            raw_text=clean,
            person=person,
            item=item,
            quantity=quantity,
            location=location,
            deadline=deadline.when if deadline else None,
            deadline_text=deadline.text if deadline else None,
            action=action,
            priority=priority,
            category=category,
            status=status,
            confidence=confidence,
        )

    def parse_many(self, updates: str | Iterable[str], reference_time: datetime | None = None) -> list[Task]:
        """Parse a newline separated log (or iterable of lines); blank lines are skipped."""
        lines = updates.splitlines() if isinstance(updates, str) else list(updates)
        tasks: list[Task] = []
        for line in lines:
            line = line.strip().lstrip("-*\u2022 ").strip()
            if not line:
                continue
            try:
                tasks.append(self.parse(line, reference_time=reference_time))
            except ValueError:
                continue
        return tasks

    def explain(self, text: str) -> str:
        """Return displaCy HTML highlighting the entities used for parsing."""
        doc = self.nlp(normalize_text(text))
        return displacy.render(doc, style="ent", jupyter=False, options={"colors": DISPLACY_COLORS})

    # -------------------------------------------------------------- extractors
    def _extract_location(self, doc: Doc) -> tuple[str | None, set[int]]:
        for ent in doc.ents:
            if ent.label_ == "LOCATION":
                return ent.text, set(range(ent.start, ent.end))
        for ent in doc.ents:
            if ent.label_ in SPACY_LOCATION_LABELS and ent.start > 0:
                prev = doc[ent.start - 1]
                if prev.lower_ in LOCATION_PREPOSITIONS:
                    return ent.text, set(range(ent.start, ent.end))
        for tok in doc:
            if (
                tok.dep_ == "pobj"
                and tok.head.lower_ in {"in", "at", "near", "inside", "outside", "behind"}
                and tok.text[:1].isupper()
                and tok.ent_type_ not in NON_ITEM_LABELS
                and tok.ent_type_ != "ITEM"
            ):
                span = doc[tok.left_edge.i : tok.right_edge.i + 1]
                return span.text, set(range(span.start, span.end))
        return None, set()

    def _looks_like_name(self, tok: Token, exclude: set[int]) -> bool:
        if tok.i in exclude:
            return False
        if tok.ent_type_ and tok.ent_type_ not in {"PERSON", "ORG", "GPE", "NORP"}:
            return False
        if tok.pos_ not in {"PROPN", "NOUN"}:
            return False
        if tok.pos_ == "NOUN" and tok.i == 0:
            return False
        if not tok.text[:1].isupper():
            return False
        return tok.lower_ not in PERSON_STOPWORDS

    def _name_span(self, doc: Doc, tok: Token) -> Span:
        start, end = tok.i, tok.i + 1
        while start > 0 and doc[start - 1].pos_ == "PROPN" and doc[start - 1].dep_ == "compound" and doc[start - 1].head == tok:
            start -= 1
        while end < len(doc) and doc[end].pos_ == "PROPN" and doc[end].dep_ in {"flat", "compound"} and doc[end].head == tok:
            end += 1
        return doc[start:end]

    def _extract_person(self, doc: Doc, exclude: set[int]) -> tuple[str | None, set[int]]:
        for ent in doc.ents:
            if ent.label_ == "PERSON" and not (set(range(ent.start, ent.end)) & exclude):
                name = ent.text.strip()
                if name.lower().endswith("'s"):
                    name = name[:-2].strip()
                return name, set(range(ent.start, ent.end))
        for tok in doc:
            if not self._looks_like_name(tok, exclude):
                continue
            head = tok.head
            is_subject = tok.dep_ in {"nsubj", "nsubjpass"} or (tok.dep_ == "poss")
            is_requested = tok.dep_ in {"dobj", "obj"} and head.lemma_.lower() in PERSON_REQUEST_VERBS
            is_prep_obj = tok.dep_ == "pobj" and head.lower_ in PERSON_PREPOSITIONS
            if is_subject or is_requested or is_prep_obj:
                span = self._name_span(doc, tok)
                return span.text, set(range(span.start, span.end))
        return None, set()

    def _is_beneficiary(self, ent: Span) -> bool:
        root = ent.root
        return root.dep_ == "pobj" and root.head.lower_ in BENEFICIARY_PREPOSITIONS

    def _bounded_span(self, doc: Doc, tok: Token, blocked: CharSpans, exclude: set[int]) -> Span:
        start, end = tok.left_edge.i, tok.right_edge.i + 1
        for i in range(tok.i + 1, end):
            t = doc[i]
            if t.pos_ in CUT_POS or i in exclude or _token_blocked(t, blocked):
                end = i
                break
        for i in range(tok.i - 1, start - 1, -1):
            t = doc[i]
            if t.pos_ in CUT_POS or t.pos_ == "CCONJ" or i in exclude or _token_blocked(t, blocked):
                start = i + 1
                break
        return doc[start:end]

    def _item_from_dependencies(self, doc: Doc, blocked: CharSpans, exclude: set[int]) -> Span | None:
        for tok in doc:
            if tok.dep_ not in {"dobj", "obj", "nsubjpass"} or tok.head.lemma_.lower() not in ACTION_VERBS:
                continue
            if tok.pos_ == "PRON" or tok.ent_type_ in NON_ITEM_LABELS or tok.i in exclude:
                continue
            return self._bounded_span(doc, tok, blocked, exclude)
        for chunk in doc.noun_chunks:
            root = chunk.root
            if root.pos_ in {"PRON", "PROPN"} or root.ent_type_ in NON_ITEM_LABELS or root.i in exclude:
                continue
            if _span_blocked(chunk, blocked):
                continue
            if root.dep_ == "nsubj" and root.head.pos_ == "VERB" and any(c.dep_ in {"dobj", "obj"} for c in root.head.children):
                continue  # subject of a transitive verb is the requester, not the item
            return self._bounded_span(doc, root, blocked, exclude)
        return None

    def _quantity_near(self, doc: Doc, start: int, end: int, blocked: CharSpans) -> int | None:
        inside = list(doc[start:end])
        before = list(reversed(doc[max(0, start - 3) : start]))
        for tok in inside + before:
            if _token_blocked(tok, blocked) or tok.ent_type_ in {"LOCATION", "TIME", "DATE", "FAC", "GPE", "LOC"}:
                continue
            value = _to_int(tok)
            if value is not None:
                return value
        return None

    def _extract_item(self, doc: Doc, blocked: CharSpans, exclude: set[int]) -> tuple[str | None, int | None]:
        item_ents = [e for e in doc.ents if e.label_ == "ITEM" and not _span_blocked(e, blocked)]
        primary = [e for e in item_ents if not self._is_beneficiary(e)]
        chosen = primary or item_ents
        if chosen:
            names = list(dict.fromkeys(e.text for e in chosen))
            quantity = self._quantity_near(doc, chosen[0].start, chosen[0].end, blocked)
            return ", ".join(names), quantity
        span = self._item_from_dependencies(doc, blocked, exclude)
        if span is None or len(span) == 0:
            return None, None
        quantity = self._quantity_near(doc, span.start, span.end, blocked)
        return _clean_item(span.text), quantity

    def _extract_action(self, doc: Doc) -> str | None:
        for tok in doc:
            lemma = tok.lemma_.lower()
            if tok.pos_ in {"VERB", "AUX", "NOUN"} and lemma in ACTION_VERBS:
                particles = [c.lower_ for c in tok.children if c.dep_ == "prt"]
                return f"{lemma} {particles[0]}" if particles else lemma
        root = next((t for t in doc if t.dep_ == "ROOT"), None)
        if root is not None and root.pos_ == "VERB":
            return root.lemma_.lower()
        return None

    # ------------------------------------------------------------- classifiers
    @staticmethod
    def _detect_status(text: str) -> str:
        if DONE_RE.search(text):
            return "done"
        if IN_PROGRESS_RE.search(text):
            return "in_progress"
        return "pending"

    @staticmethod
    def _detect_priority(text: str, deadline: DeadlineMatch | None, reference: datetime) -> str:
        if HIGH_PRIORITY_RE.search(text) or (deadline and deadline.kind == "asap"):
            return "high"
        if LOW_PRIORITY_RE.search(text):
            return "low"
        if deadline and deadline.when is not None and deadline.when - reference <= timedelta(minutes=30):
            return "high"
        return "medium"

    @staticmethod
    def _confidence(person: str | None, item: str | None, deadline: DeadlineMatch | None, location: str | None) -> float:
        score = 0.1
        score += 0.4 if item else 0.0
        score += 0.2 if person else 0.0
        score += 0.2 if deadline else 0.0
        score += 0.1 if location else 0.0
        return round(min(score, 1.0), 2)
