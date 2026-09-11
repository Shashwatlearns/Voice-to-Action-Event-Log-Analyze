"""Voice-to-Action Event Log Analyzer.

Parses spoken or typed event updates such as ``"Rahul needs a VGA cable by 4 PM"``
into structured, actionable tasks using spaCy Named Entity Recognition.
"""

from voice_to_action.models import Task
from voice_to_action.parser import EventLogParser, ModelNotFoundError
from voice_to_action.storage import TaskStore

__version__ = "0.1.0"
__all__ = ["EventLogParser", "ModelNotFoundError", "Task", "TaskStore", "__version__"]
