# Voice-to-Action Event Log Analyzer

An NLP tool that turns spoken or typed live-event updates into an actionable task table.

> "Rahul needs a VGA cable by 4 PM" -> **Rahul** | **VGA cable** | **16:00** | av_equipment | medium

During live events, logistics requests fly around over radios, WhatsApp and shouted hallway conversations, and
things get missed. This tool uses **Named Entity Recognition** (spaCy) plus deterministic rules to extract
**who** needs **what**, **how many**, **where** and **by when**, then tracks it all on a live task board with
overdue / due-soon highlighting and CSV export.

## Features

- **NER parsing** with spaCy + a domain `EntityRuler` (200+ event items and locations such as `VGA cable`,
  `lapel mic`, `Hall B`, `registration desk`) and dependency-parse fallbacks for unknown items.
- **Deadline resolution** for `by 4 PM`, `at 16:30`, `before noon`, `in 20 minutes`, `tomorrow 9am`,
  `by end of day`, `asap`, with reference-time anchoring so results are reproducible.
- **Extracted fields**: person, item, quantity (digits and words), location, deadline, action verb, priority
  (urgent words or tight deadlines), category (AV, furniture, catering, personnel, power, signage, ...),
  status (`pending` / `in_progress` / `done` inferred from phrases like "is delivered"), confidence score.
- **Voice input**: microphone recording or audio upload via `SpeechRecognition` (Google Web Speech) or offline
  Whisper backends. Typed input always works without audio dependencies.
- **Streamlit task board**: editable table, filters, overdue / due-soon flags, entity highlighting, per-person
  and per-category charts, CSV / JSON export, SQLite persistence.
- **CLI** (`vta`) for scripting and batch files, **tests**, **GitLab CI**, **Dockerfile**.

## Architecture

```
            +-----------------+       +------------------------+
 voice ---> | voice.py        | text  | parser.py              |
 (mic/file) | SpeechRecogn. / | ----> |  normalise             |       +-------------+
            | Whisper         |       |  deadline.py (regex)   | Task  | storage.py  | CSV/JSON
 typed ---------------------------->  |  spaCy EntityRuler+NER | ----> | SQLite      | ------->
                                      |  dependency heuristics |       +------+------+
                                      |  priority/category     |              |
                                      +------------------------+       +------v------+
                                                                       | app.py (UI) |
                                                                       | cli.py      |
                                                                       +-------------+
```

| Module | Responsibility |
| --- | --- |
| `src/voice_to_action/parser.py` | `EventLogParser`: orchestrates NER, dependency heuristics and classifiers |
| `src/voice_to_action/deadline.py` | Deterministic deadline phrase extraction and resolution |
| `src/voice_to_action/patterns.py` | Domain vocabulary: items, locations, categories, verbs, priority regexes |
| `src/voice_to_action/models.py` | `Task` dataclass, due-state logic, (de)serialisation |
| `src/voice_to_action/storage.py` | `TaskStore`: SQLite CRUD, filters, CSV / JSON export |
| `src/voice_to_action/voice.py` | Speech-to-text backends (all optional) |
| `src/voice_to_action/cli.py` | `vta` command line interface |
| `app.py` | Streamlit UI |

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# core + UI + voice (Google Web Speech backend)
pip install -e ".[ui,voice,dev]"
python -m spacy download en_core_web_sm

cp .env.example .env                 # optional: tweak paths / backends
```

Optional extras:

| Extra | Adds | Notes |
| --- | --- | --- |
| `voice` | `SpeechRecognition`, `pydub` | Google Web Speech backend (online). Install **ffmpeg** to accept mp3/m4a/ogg uploads. |
| `whisper` | `faster-whisper` | Offline `faster-whisper` backend, no API calls |
| `mic` | `PyAudio` | Needed only for the CLI `--listen` microphone mode (the Streamlit recorder works in-browser) |

For the `whisper` backend via SpeechRecognition install `openai-whisper` manually.

## Run the Streamlit app

```bash
streamlit run app.py
```

1. **Log updates**: record with the in-browser microphone, upload audio, or type one update per line
   (click *Load sample updates* to try it). Press **Parse & add to board**.
2. **Task board**: edit any cell inline, change status, tick *Delete?*, then **Apply changes**. Export CSV / JSON.
3. **Insights**: next deadlines, load per person and per category, priority mix.

The sidebar lets you set the **reference time** (useful when replaying a log), the *due soon* window, the
speech backend and board filters. Tasks persist in `data/tasks.db` (`VTA_DB_PATH`).

## CLI usage

```bash
# parse inline updates
vta "Rahul needs a VGA cable by 4 PM" "Priya wants 20 chairs in Hall B before 6:30"

# batch file -> CSV / JSON, anchored to a fixed reference time
vta --file examples/sample_updates.txt --ref "2026-09-11 14:00" --csv tasks.csv --json tasks.json

# transcribe an audio file, persist to the shared task board used by the UI
vta --audio update.wav --backend google --save

# listen on the microphone (needs the `mic` extra)
vta --listen --backend faster-whisper

# pipe input
cat updates.txt | vta
```

`python -m voice_to_action ...` is equivalent to `vta ...`.

Example output:

```
                                   Actionable tasks
 Person  Item             Qty  Location           Deadline      Priority  Category       Status   Conf
 Rahul   VGA cable        -    -                  11 Sep 16:00  medium    av_equipment   pending  0.90
 Priya   chairs           20   Hall B             11 Sep 18:30  medium    furniture      pending  1.00
 -       projector        -    main stage         11 Sep 12:00  medium    av_equipment   pending  0.80
```

## Python API

```python
from datetime import datetime
from voice_to_action import EventLogParser, TaskStore

parser = EventLogParser()
task = parser.parse("Rahul needs a VGA cable by 4 PM", reference_time=datetime(2026, 9, 11, 14, 0))
print(task.person, task.item, task.deadline, task.category)   # Rahul VGA cable 2026-09-11 16:00:00 av_equipment

store = TaskStore("data/tasks.db")
store.add(task)
print(store.export_csv())
```

## Voice backend configuration

| `VTA_VOICE_BACKEND` | Requirements | Online? |
| --- | --- | --- |
| `google` (default) | `voice` extra | yes (free Google Web Speech endpoint) |
| `whisper` | `voice` extra + `pip install openai-whisper` | no |
| `faster-whisper` | `whisper` extra | no |

Set `VTA_WHISPER_MODEL` (`tiny`, `base`, `small`, ...) to trade accuracy for speed on the Whisper backends.

## How parsing works

1. **Normalise** whitespace and quotes.
2. **Deadline** phrase is located with regexes (`deadline.py`) and resolved against the reference time.
   Bare hours (`by 4`) are disambiguated to the next plausible occurrence; `asap` becomes *now + 15 min* and
   forces high priority.
3. **spaCy pipeline**: an `EntityRuler` (labels `ITEM`, `LOCATION`) runs *before* the statistical NER so domain
   terms win; NER still supplies `PERSON`, `TIME`, `CARDINAL`, `FAC`, ...
4. **Heuristics** fill gaps: subject of the request verb -> person; object of the verb -> item; number directly
   before the item -> quantity; `for the projector` is treated as a beneficiary, not the item.
5. **Classifiers**: keyword categories, priority from urgency words or time-to-deadline, status from phrases
   such as "is delivered" or "working on it".
6. A **confidence** score (0-1) reflects how many of person / item / deadline / location were found.

Extend the vocabulary by editing `ITEM_PHRASES`, `LOCATION_PHRASES` and `CATEGORY_KEYWORDS` in
`src/voice_to_action/patterns.py`.

## Tests and linting

```bash
pytest                      # parser tests skip automatically if the spaCy model is missing
VTA_REQUIRE_MODEL=1 pytest  # fail instead of skip (used in CI)
ruff check .
```

The GitLab pipeline runs `ruff`, `pytest` with coverage and JUnit reports, and builds the Docker image on the
default branch.

## Docker

```bash
docker build -t voice-to-action .
docker run -p 8501:8501 -v $(pwd)/data:/data voice-to-action
```

Open http://localhost:8501.

## Roadmap ideas

- Multi-item updates split into separate tasks ("a projector and 2 mics")
- Assignee vs requester distinction ("ask Rahul to bring...")
- Push notifications / Slack webhook for overdue tasks
- Fine-tuned NER model trained on real event logs

## License

MIT
