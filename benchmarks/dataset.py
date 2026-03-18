"""Load benchmark dataset from data.json — single source of truth shared with Go."""

import json
from pathlib import Path

_DATA_FILE = Path(__file__).parent / "data.json"

with open(_DATA_FILE) as f:
    _raw = json.load(f)

# Expose as list of tuples: (cached_key, lookup_query, answer, expect_hit, category)
DATASET = [(d["cached"], d["lookup"], d["answer"], d["expect_hit"], d["category"]) for d in _raw]
