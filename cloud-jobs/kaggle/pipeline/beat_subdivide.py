"""Split long beats into shorter visual segments (~4s each)."""

import copy
import math


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

        for part in range(n):
            sub = copy.deepcopy(beat)
            sub["sceneIndex"] = new_idx
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
                    "narrationText": beat.get("narrationText", ""),
                }
            )
            new_idx += 1

    return new_beats, new_timings
