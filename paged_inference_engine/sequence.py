"""Sequence state. Pure Python, fully testable on your laptop."""

from __future__ import annotations

import enum
import itertools
from dataclasses import dataclass, field
from typing import Optional

from .sampling import SamplingParams


class SequenceStatus(enum.Enum):
    WAITING = enum.auto()
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()
    FINISHED_STOPPED = enum.auto()      # hit EOS or a stop token
    FINISHED_LENGTH = enum.auto()       # hit max_tokens
