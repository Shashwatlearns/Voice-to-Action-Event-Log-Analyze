"""Command line interface: ``python -m voice_to_action ...`` or the ``vta`` script."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from voice_to_action import config
from voice_to_action.models import Task
from voice_to_action.parser import EventLogParser, ModelNotFoundError
from voice_to_action.storage import TaskStore
from voice_to_action.voice import VOICE_BACKENDS, VoiceUnavailableError, transcribe_audio, transcribe_microphone


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vta",
        description="Voice-to-Action Event Log Analyzer: turn event updates into an actionable task table.",
    )
    source = parser.add_argument_group("input (choose one or more)")
    source.add_argument("text", nargs="*", help="one or more updates, e.g. \"Rahul needs a VGA cable by 4 PM\"")
    source.add_argument("-f", "--file", type=Path, help="text file with one update per line")
    source.add_argument("-a", "--audio", type=Path, help="audio file to transcribe (wav/mp3/m4a/flac)")
    source.add_argument("-l", "--listen", action="store_true", help="record one phrase from the microphone")

    parser.add_argument(
        "--backend", choices=VOICE_BACKENDS, default=config.VOICE_BACKEND, help="speech-to-text backend"
    )
    parser.add_argument("--ref", help="reference time for relative deadlines, e.g. '2026-09-11 14:00' (default: now)")
    parser.add_argument("--csv", type=Path, help="write parsed tasks to this CSV file")
    parser.add_argument("--json", type=Path, help="write parsed tasks to this JSON file")
    parser.add_argument("--save", action="store_true", help="persist tasks to the SQLite task board")
    parser.add_argument(
        "--db", default=config.DB_PATH, help=f"SQLite path used with --save (default: {config.DB_PATH})"
    )
    parser.add_argument("--model", default=config.SPACY_MODEL, help="spaCy model name")
    return parser


def _collect_text(args: argparse.Namespace) -> list[str]:
    lines: list[str] = list(args.text or [])
    if args.file:
        lines.extend(args.file.read_text(encoding="utf-8").splitlines())
    if args.audio:
        transcript = transcribe_audio(args.audio, backend=args.backend)
        print(f"[transcript] {transcript}", file=sys.stderr)
        lines.append(transcript)
    if args.listen:
        print("Listening... speak your update now.", file=sys.stderr)
        transcript = transcribe_microphone(backend=args.backend)
        print(f"[transcript] {transcript}", file=sys.stderr)
        lines.append(transcript)
    if not lines and not sys.stdin.isatty():
        lines.extend(sys.stdin.read().splitlines())
    return lines


def _print_table(tasks: list[Task]) -> None:
    try:
        from rich.console import Console
        from rich.table import Table
    except ImportError:  # pragma: no cover - rich is a core dependency
        for task in tasks:
            print(task.summary())
        return

    table = Table(title="Actionable tasks", show_lines=False, expand=True)
    for column in ("Person", "Item", "Qty", "Location", "Deadline", "Priority", "Category", "Status", "Conf"):
        table.add_column(column)
    colours = {"high": "bold red", "medium": "yellow", "low": "green"}
    for task in tasks:
        table.add_row(
            task.person or "-",
            task.item or "-",
            str(task.quantity) if task.quantity else "-",
            task.location or "-",
            task.deadline.strftime("%d %b %H:%M") if task.deadline else (task.deadline_text or "-"),
            f"[{colours[task.priority]}]{task.priority}[/]",
            task.category,
            task.status,
            f"{task.confidence:.2f}",
        )
    Console().print(table)


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        lines = _collect_text(args)
    except VoiceUnavailableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not lines:
        build_arg_parser().print_help()
        return 1

    reference = datetime.fromisoformat(args.ref) if args.ref else None
    try:
        parser = EventLogParser(model_name=args.model)
    except ModelNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    tasks = parser.parse_many(lines, reference_time=reference)
    if not tasks:
        print("No actionable updates found.", file=sys.stderr)
        return 1
    _print_table(tasks)

    store = TaskStore(args.db if args.save else ":memory:")
    if args.save:
        store.add_many(tasks)
        print(f"Saved {len(tasks)} task(s) to {args.db}", file=sys.stderr)
    if args.csv:
        store.export_csv(args.csv, tasks=tasks)
        print(f"Wrote {args.csv}", file=sys.stderr)
    if args.json:
        store.export_json(args.json, tasks=tasks)
        print(f"Wrote {args.json}", file=sys.stderr)
    store.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
