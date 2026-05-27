"""Validation and retry helpers."""

import os
import time

from .logging_util import log_stage


def retry_stage(fn, stage, record_id, max_attempts=3, delay_sec=2.0):
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as e:
            last_err = e
            log_stage(
                stage,
                record_id,
                message=f"attempt {attempt}/{max_attempts} failed: {e}",
                level="ERROR",
            )
            if attempt < max_attempts:
                time.sleep(delay_sec * attempt)
    raise last_err


def validate_beats(beats):
    if not beats or not isinstance(beats, list):
        raise ValueError("beats missing or empty")
    for i, b in enumerate(beats):
        if not str(b.get("narrationText", "")).strip() and not str(
            b.get("visualPrompt", "")
        ).strip():
            raise ValueError(f"beat {i} has no narration or visual prompt")


def validate_audio(path, min_duration=0.5):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"audio missing: {path}")
    if os.path.getsize(path) < 1000:
        raise ValueError(f"audio too small: {path}")


def validate_scene_clips(paths):
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f"scene clips missing: {missing[:3]}")


def validate_video(path, min_bytes=5000):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"video missing: {path}")
    if os.path.getsize(path) < min_bytes:
        raise ValueError(f"video too small: {path}")
