"""Streamlit UI for the Voice-to-Action Event Log Analyzer.

Run with: ``streamlit run app.py``
"""

from __future__ import annotations

import sys
from datetime import datetime, time
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from voice_to_action import config  # noqa: E402
from voice_to_action.models import PRIORITIES, STATUSES, Task  # noqa: E402
from voice_to_action.parser import EventLogParser, ModelNotFoundError  # noqa: E402
from voice_to_action.patterns import CATEGORY_KEYWORDS  # noqa: E402
from voice_to_action.storage import TaskStore  # noqa: E402
from voice_to_action.voice import VOICE_BACKENDS, VoiceUnavailableError, transcribe_audio  # noqa: E402

st.set_page_config(page_title="Voice-to-Action Event Log Analyzer", page_icon="\U0001f399\ufe0f", layout="wide")

DUE_LABELS = {
    "overdue": "\U0001f534 Overdue",
    "due_soon": "\U0001f7e0 Due soon",
    "on_track": "\U0001f7e2 On track",
    "done": "\u2705 Done",
    "none": "\u26aa No deadline",
}
CATEGORIES = [*CATEGORY_KEYWORDS.keys(), "other"]
BOARD_COLUMNS = ["id", "Due", "Person", "Item", "Qty", "Location", "Deadline", "Priority", "Category", "Status", "Confidence", "Raw", "Delete"]
SAMPLE_FILE = Path(__file__).resolve().parent / "examples" / "sample_updates.txt"


# ----------------------------------------------------------------------------- resources
@st.cache_resource(show_spinner="Loading spaCy model...")
def get_parser() -> EventLogParser:
    return EventLogParser()


@st.cache_resource
def get_store() -> TaskStore:
    return TaskStore(config.DB_PATH)


def tasks_to_frame(tasks: list[Task], now: datetime, soon_minutes: int) -> pd.DataFrame:
    rows = [
        {
            "id": t.id,
            "Due": DUE_LABELS[t.due_state(now, soon_minutes)],
            "Person": t.person or "",
            "Item": t.item or "",
            "Qty": t.quantity,
            "Location": t.location or "",
            "Deadline": t.deadline,
            "Priority": t.priority,
            "Category": t.category,
            "Status": t.status,
            "Confidence": t.confidence,
            "Raw": t.raw_text,
            "Delete": False,
        }
        for t in tasks
    ]
    frame = pd.DataFrame(rows, columns=BOARD_COLUMNS)
    frame["Deadline"] = pd.to_datetime(frame["Deadline"], errors="coerce")
    frame["Qty"] = frame["Qty"].astype("Int64")
    return frame


def _clean_str(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


def persist_edits(store: TaskStore, edited: pd.DataFrame) -> tuple[int, int]:
    """Write user edits from the data editor back to the store. Returns (updated, deleted)."""
    updated = deleted = 0
    for _, row in edited.iterrows():
        task = store.get(str(row["id"]))
        if task is None:
            continue
        if bool(row["Delete"]):
            store.delete(task.id)
            deleted += 1
            continue
        deadline = None if pd.isna(row["Deadline"]) else pd.Timestamp(row["Deadline"]).to_pydatetime().replace(tzinfo=None)
        quantity = None if pd.isna(row["Qty"]) else int(row["Qty"])
        changes = {
            "person": _clean_str(row["Person"]),
            "item": _clean_str(row["Item"]),
            "quantity": quantity,
            "location": _clean_str(row["Location"]),
            "deadline": deadline,
            "priority": str(row["Priority"]),
            "category": str(row["Category"]),
            "status": str(row["Status"]),
        }
        diff = {k: v for k, v in changes.items() if getattr(task, k) != v}
        if diff:
            store.update_fields(task.id, **diff)
            updated += 1
    return updated, deleted


# ------------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("\U0001f399\ufe0f Voice-to-Action")
    st.caption("Event Log Analyzer")

    st.subheader("Reference time")
    use_now = st.toggle("Use current time", value=True, help="Relative deadlines like 'in 20 minutes' are resolved against this.")
    if use_now:
        reference_time = datetime.now()
    else:
        ref_date = st.date_input("Event date", value=datetime.now().date())
        ref_clock = st.time_input("Current time", value=datetime.now().time().replace(second=0, microsecond=0))
        reference_time = datetime.combine(ref_date, ref_clock or time(9, 0))
    soon_minutes = st.slider("'Due soon' window (minutes)", 5, 120, config.DUE_SOON_MINUTES, step=5)

    st.subheader("Voice")
    backend = st.selectbox("Speech-to-text backend", VOICE_BACKENDS, index=VOICE_BACKENDS.index(config.VOICE_BACKEND) if config.VOICE_BACKEND in VOICE_BACKENDS else 0)

    st.subheader("Board filters")
    status_filter = st.multiselect("Status", STATUSES, default=list(STATUSES))
    priority_filter = st.multiselect("Priority", PRIORITIES, default=list(PRIORITIES))
    category_filter = st.multiselect("Category", CATEGORIES, default=CATEGORIES)
    person_filter = st.text_input("Person contains")

    st.divider()
    st.caption(f"Database: `{config.DB_PATH}`")

# ------------------------------------------------------------------------------ resources
try:
    parser = get_parser()
except ModelNotFoundError as exc:
    st.error(str(exc))
    st.stop()
store = get_store()

st.title("Voice-to-Action Event Log Analyzer")
st.caption("Speak or type live-event updates; NER turns them into an actionable task table so nothing gets missed.")

tab_log, tab_board, tab_insights = st.tabs(["\U0001f4dd Log updates", "\U0001f4cb Task board", "\U0001f4ca Insights"])

# ---------------------------------------------------------------------------- tab: log
with tab_log:
    col_voice, col_text = st.columns([1, 1.4])

    with col_voice:
        st.subheader("Voice input")
        audio_bytes: bytes | None = None
        audio_name = "recording.wav"
        if hasattr(st, "audio_input"):
            recording = st.audio_input("Record an update")
            if recording is not None:
                audio_bytes, audio_name = recording.getvalue(), "recording.wav"
        uploaded = st.file_uploader("...or upload an audio file", type=["wav", "mp3", "m4a", "flac", "ogg"])
        if uploaded is not None:
            audio_bytes, audio_name = uploaded.getvalue(), uploaded.name
        if st.button("Transcribe", disabled=audio_bytes is None, use_container_width=True):
            try:
                with st.spinner(f"Transcribing with {backend}..."):
                    transcript = transcribe_audio(audio_bytes or b"", backend=backend, filename=audio_name)
                existing = st.session_state.get("update_text", "").rstrip()
                st.session_state["update_text"] = f"{existing}\n{transcript}".strip()
                st.success("Transcript added to the text box. Review it, then parse.")
            except (VoiceUnavailableError, ValueError) as exc:
                st.error(str(exc))
        if st.button("Load sample updates", use_container_width=True) and SAMPLE_FILE.exists():
            st.session_state["update_text"] = SAMPLE_FILE.read_text(encoding="utf-8").strip()

    with col_text:
        st.subheader("Typed input")
        update_text = st.text_area(
            "One update per line",
            key="update_text",
            height=220,
            placeholder="Rahul needs a VGA cable by 4 PM\nPriya wants 20 chairs in Hall B before 6:30",
        )
        parse_col, clear_col = st.columns([1, 1])
        parse_clicked = parse_col.button("Parse & add to board", type="primary", use_container_width=True, disabled=not update_text.strip())
        if clear_col.button("Clear text", use_container_width=True):
            st.session_state["update_text"] = ""
            st.rerun()

    if parse_clicked:
        new_tasks = parser.parse_many(update_text, reference_time=reference_time)
        if not new_tasks:
            st.warning("Nothing actionable found in the text.")
        else:
            store.add_many(new_tasks)
            st.session_state["last_parsed"] = [t.id for t in new_tasks]
            st.success(f"Added {len(new_tasks)} task(s) to the board.")

    last_ids = st.session_state.get("last_parsed", [])
    last_tasks = [t for t in (store.get(i) for i in last_ids) if t is not None]
    if last_tasks:
        st.subheader("Last parsed")
        preview = tasks_to_frame(last_tasks, reference_time, soon_minutes).drop(columns=["id", "Delete", "Raw"])
        st.dataframe(preview, use_container_width=True, hide_index=True)
        with st.expander("Show entity highlighting"):
            for task in last_tasks:
                st.markdown(f"**{task.summary()}**  \n`confidence {task.confidence:.2f}`")
                components.html(parser.explain(task.raw_text), height=70, scrolling=False)

# -------------------------------------------------------------------------- tab: board
with tab_board:
    tasks = store.list_tasks(status=status_filter or None, priority=priority_filter or None, category=category_filter or None, person=person_filter or None)
    board = tasks_to_frame(tasks, reference_time, soon_minutes)

    overdue = sum(t.due_state(reference_time, soon_minutes) == "overdue" for t in tasks)
    due_soon = sum(t.due_state(reference_time, soon_minutes) == "due_soon" for t in tasks)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pending", sum(t.status != "done" for t in tasks))
    m2.metric("Overdue", overdue)
    m3.metric("Due soon", due_soon)
    m4.metric("Done", sum(t.status == "done" for t in tasks))

    if board.empty:
        st.info("No tasks yet. Log some updates in the first tab.")
    else:
        edited = st.data_editor(
            board,
            key="board_editor",
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            column_order=[c for c in BOARD_COLUMNS if c != "id"],
            disabled=["Due", "Confidence", "Raw"],
            column_config={
                "Due": st.column_config.TextColumn("Due", width="small"),
                "Qty": st.column_config.NumberColumn("Qty", min_value=0, step=1, width="small"),
                "Deadline": st.column_config.DatetimeColumn("Deadline", format="DD MMM HH:mm", step=60),
                "Priority": st.column_config.SelectboxColumn("Priority", options=list(PRIORITIES), width="small"),
                "Category": st.column_config.SelectboxColumn("Category", options=CATEGORIES),
                "Status": st.column_config.SelectboxColumn("Status", options=list(STATUSES), width="small"),
                "Confidence": st.column_config.ProgressColumn("Conf.", min_value=0.0, max_value=1.0, format="%.2f", width="small"),
                "Raw": st.column_config.TextColumn("Original update", width="large"),
                "Delete": st.column_config.CheckboxColumn("Delete?", width="small"),
            },
        )
        b1, b2, b3, b4 = st.columns(4)
        if b1.button("Apply changes", type="primary", use_container_width=True):
            n_upd, n_del = persist_edits(store, edited)
            st.toast(f"Updated {n_upd}, deleted {n_del}")
            st.rerun()
        b2.download_button(
            "Export CSV",
            data=store.export_csv(tasks=tasks),
            file_name=f"event_tasks_{reference_time:%Y%m%d_%H%M}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        b3.download_button(
            "Export JSON",
            data=store.export_json(tasks=tasks),
            file_name=f"event_tasks_{reference_time:%Y%m%d_%H%M}.json",
            mime="application/json",
            use_container_width=True,
        )
        if b4.button("Clear done tasks", use_container_width=True):
            removed = store.clear_done()
            st.toast(f"Removed {removed} completed task(s)")
            st.rerun()

# ----------------------------------------------------------------------- tab: insights
with tab_insights:
    all_tasks = store.list_tasks()
    if not all_tasks:
        st.info("Insights appear once tasks have been logged.")
    else:
        frame = tasks_to_frame(all_tasks, reference_time, soon_minutes)
        open_frame = frame[frame["Status"] != "done"]

        st.subheader("Next up")
        upcoming = open_frame.dropna(subset=["Deadline"]).sort_values("Deadline").head(8)
        if upcoming.empty:
            st.caption("No open tasks with deadlines.")
        else:
            for _, row in upcoming.iterrows():
                qty = f"{int(row['Qty'])}x " if pd.notna(row["Qty"]) else ""
                where = f" @ {row['Location']}" if row["Location"] else ""
                who = row["Person"] or "unassigned"
                st.markdown(f"- {row['Due']} **{row['Deadline']:%H:%M}** {qty}{row['Item'] or '?'}{where} ({who}, {row['Priority']})")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Open tasks per person")
            per_person = open_frame.assign(Person=open_frame["Person"].replace("", "unassigned")).groupby("Person").size().sort_values(ascending=False)
            st.bar_chart(per_person)
        with c2:
            st.subheader("Open tasks per category")
            st.bar_chart(open_frame.groupby("Category").size().sort_values(ascending=False))

        st.subheader("Priority mix")
        st.bar_chart(frame.groupby(["Priority", "Status"]).size().unstack(fill_value=0))
