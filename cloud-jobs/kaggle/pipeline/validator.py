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


def _norm_name(value):
    return str(value or "").strip().casefold()


def validate_characters(characters, beats):
    """Ensure job has script-derived characters and beat references are valid."""
    if not characters or not isinstance(characters, list):
        raise ValueError("characters missing or empty — run script extraction")

    registry_by_name = {}
    registry_by_id = {}
    for c in characters:
        name = _norm_name(c.get("name"))
        cid = str(c.get("id") or "").strip().casefold()
        if name:
            registry_by_name[name] = c
        if cid:
            registry_by_id[cid] = c
        if not str(c.get("referencePrompt") or "").strip():
            raise ValueError(f"character missing referencePrompt: {c.get('name')}")

    for i, beat in enumerate(beats or []):
        names = beat.get("characters") or []
        if not names:
            raise ValueError(f"beat {i} has no characters assigned")
        for n in names:
            key = _norm_name(n)
            if key not in registry_by_name:
                raise ValueError(f"beat {i} unknown character: {n}")
