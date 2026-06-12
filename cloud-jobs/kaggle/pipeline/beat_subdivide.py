"""Split long beats into shorter visual segments (~4s each)."""

import copy
import math
import re

from .beat_metadata import refresh_beat_from_narration


def _split_to_n_parts(text, n):
    """
    Split narration text into n roughly equal parts using sentence boundaries.
    Hindi sentence boundary: ।  English: . ! ?
    Falls back to word-chunk split when fewer than 2 sentences are found.
    Each part is non-empty — falls back to the full text for empty slots.
    """
    text = str(text or "").strip()
    if n <= 1 or not text:
        return [text] * max(1, n)

    sentences = [s.strip() for s in re.split(r"(?<=[।.!?])\s+", text) if s.strip()]

    if len(sentences) < 2:
        # No sentence boundaries found — distribute word chunks
        words = text.split()
        if not words:
            return [text] * n
        per = max(1, len(words) // n)
        parts = []
        for i in range(n):
            start = i * per
            end = (i + 1) * per if i < n - 1 else len(words)
            chunk = " ".join(words[start:end]).strip()
            parts.append(chunk or text)
        return parts

    per = max(1, len(sentences) // n)
    parts = []
    for i in range(n):
        start = i * per
        end = (i + 1) * per if i < n - 1 else len(sentences)
        chunk = " ".join(sentences[start:end]).strip()
        parts.append(chunk or text)
    return parts


def subdivide_long_beats(beats, timings, max_sec=4.5):
    """
    Returns (new_beats, new_timings) with one visual segment per max_sec of audio.
    TTS is already baked into master audio; this only splits visual/timing slots.
    """
    timing_by_idx = {t["sceneIndex"]: t for t in (timings or [])}
    new_beats = []
    new_timings = []
    new_idx = 0
    max_sec = max(2.0, float(max_sec))

    for beat in beats or []:
        idx = beat.get("sceneIndex", new_idx)
        t = timing_by_idx.get(idx, {})
        dur = float(t.get("duration") or beat.get("duration") or max_sec)
        n = max(1, int(math.ceil(dur / max_sec)))
        sub_dur = dur / n

        # Split narration text into n sentence-based slices so each sub-beat
        # gets a unique excerpt — image pipeline will use this to generate a
        # scene that visually matches what is being spoken at that moment.
        text_parts = _split_to_n_parts(beat.get("narrationText", ""), n)

        for part in range(n):
            sub = copy.deepcopy(beat)
            sub["sceneIndex"] = new_idx

            # Assign the sentence slice for this sub-beat
            sub["narrationText"] = text_parts[part]

            # Clear pre-computed visualPrompt for sub-beats (n > 1) so
            # generate_scene_image builds a fresh prompt from the new slice.
            # For n == 1 the original visualPrompt is preserved.
            if n > 1:
                sub["visualPrompt"] = ""
                refresh_beat_from_narration(sub)

            if part > 0:
                sub["beatTitle"] = f"{beat.get('beatTitle', 'Beat')} ({part + 1}/{n})"
            new_beats.append(sub)

            start = float(t.get("start", 0)) + part * sub_dur
            end = start + sub_dur
            new_timings.append(
                {
                    "sceneIndex": new_idx,
                    "start": start,
                    "end": end,
                    "duration": sub_dur,
                    "narrationText": text_parts[part],
                    "action": str(sub.get("action", "")).strip(),
                    "actionPose": str(sub.get("actionPose") or sub.get("action", "")).strip(),
                    "cameraStyle": str(sub.get("cameraStyle", "")).strip(),
                    "emotion": str(sub.get("emotion", "neutral")).strip(),
                    "environment": str(sub.get("environment", "")).strip(),
                    "propsInFrame": sub.get("propsInFrame") or [],
                }
            )
            new_idx += 1

    return new_beats, new_timings
