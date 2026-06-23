"""Script-only scene action resolution (mirrors lib/videoPipeline/actionPose.js)."""

import re

_GENERIC_ACTIONS = {
    "",
    "story moment",
    "narration",
    "story scene",
    "neutral",
    "neutral scene",
    "character scene",
    "key story moment, clear visible action in frame",
    "clear frozen story moment, expressive body language visible",
}

_GENERIC_POSE_PREFIXES = (
    "standing proudly",
    "walking along a path",
    "looking around cautiously",
    "sitting and listening",
    "talking expressively",
    "speaking expressively",
)

_TEMPLATE_ACTION_PATTERNS = (
    re.compile(r"fighting dynamically", re.I),
    re.compile(r"speaking expressively", re.I),
    re.compile(r"talking expressively", re.I),
    re.compile(r"standing proudly", re.I),
    re.compile(r"clear frozen story moment", re.I),
    re.compile(r"key story moment", re.I),
    re.compile(r"combat pose with weapon", re.I),
)


def _is_generic_action(value):
    v = str(value or "").strip().lower()
    if not v:
        return True
    if v in _GENERIC_ACTIONS:
        return True
    if re.match(r"^(standing|sitting|walking|looking|character scene|story scene)\b", v):
        return True
    if any(v.startswith(p) for p in _GENERIC_POSE_PREFIXES):
        return True
    return any(p.search(v) for p in _TEMPLATE_ACTION_PATTERNS)


def _pick_script_field(beat, fields):
    for key in fields:
        val = str(beat.get(key) or "").strip()
        if val and not _is_generic_action(val):
            return val
    return ""


def extract_story_action(beat, previous_beat=None):
    """Return action from beat script fields only — no keyword rules."""
    del previous_beat
    return _pick_script_field(
        beat, ("action", "scriptEvent", "summary", "narrationText")
    )


def resolve_scene_action(beat, previous_beat=None):
    """Trust job payload script fields; never inject templates."""
    del previous_beat
    action = _pick_script_field(beat, ("actionPose", "action", "scriptEvent", "summary"))
    if action:
        return action
    title = str(beat.get("beatTitle") or "").strip()
    if title and not _is_generic_action(title):
        return title
    return str(beat.get("narrationText") or "").strip()
