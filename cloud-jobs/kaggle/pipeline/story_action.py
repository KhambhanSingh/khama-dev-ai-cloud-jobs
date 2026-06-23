"""Script-only scene action resolution (mirrors lib/videoPipeline/actionPose.js)."""

from .prompt_sanitize import pick_english_beat_line, strip_forbidden_prompt_words


def _is_generic_action(value):
    v = str(value or "").strip().lower()
    if not v:
        return True
    markers = (
        "story moment",
        "key story moment",
        "clear frozen story moment",
        "story scene",
    )
    return any(m in v for m in markers)


def extract_story_action(beat, previous_beat=None):
    """Return English action from beat production fields only."""
    del previous_beat
    return pick_english_beat_line(beat)


def resolve_scene_action(beat, previous_beat=None):
    """Trust job payload — English fields only, never Hindi narration."""
    del previous_beat
    for key in ("actionPose", "action", "scriptEvent", "summary", "visualPrompt", "beatTitle"):
        val = strip_forbidden_prompt_words(str(beat.get(key) or "").strip())
        if val and not _is_generic_action(val):
            from .prompt_sanitize import is_english_prompt_text

            if is_english_prompt_text(val):
                return val
    return pick_english_beat_line(beat)
