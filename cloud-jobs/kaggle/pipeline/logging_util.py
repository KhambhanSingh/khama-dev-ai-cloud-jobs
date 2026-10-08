"""Structured stage logging for Kaggle worker."""

import sys
from datetime import datetime


def log_stage(stage, record_id="", beat=None, message="", level="INFO"):
    ts = datetime.utcnow().strftime("%H:%M:%S")
    parts = [f"[{ts}]", f"STAGE={stage}", f"record={record_id}"]
    if beat is not None:
        parts.append(f"beat={beat}")
    if message:
        parts.append(str(message))
    line = " ".join(parts)
    print(line, flush=True)
    if level == "ERROR":
        print(line, file=sys.stderr, flush=True)
