"""Turn results/*.json into the figures that go at the top of the README.

The headline figure is a THROUGHPUT vs P99-LATENCY curve, one line per system, points
labelled by request rate. Not a bar chart. A bar chart of "tokens/sec" hides the
operating point and invites the one question you cannot then answer: "at what latency?"

    python -m bench.plot --results results --out docs/figures
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
