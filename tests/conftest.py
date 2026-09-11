from __future__ import annotations

import os
from datetime import datetime

import pytest

from voice_to_action.parser import EventLogParser, ModelNotFoundError

REF = datetime(2026, 9, 11, 14, 0)


@pytest.fixture(scope="session")
def parser() -> EventLogParser:
    try:
        return EventLogParser()
    except ModelNotFoundError as exc:
        if os.environ.get("VTA_REQUIRE_MODEL"):
            raise
        pytest.skip(str(exc))


@pytest.fixture
def ref() -> datetime:
    return REF
