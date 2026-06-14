#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

"""Trace verification command for final clean release."""
from htce_origin.governance.evidence import TraceLog

log = TraceLog()
event = log.append("verify_trace_release", {"ok": True})
print(event.event_hash())
