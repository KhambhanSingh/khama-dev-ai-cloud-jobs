"""Refresh beat action/emotion/environment from a narration slice (no torch)."""

from .environments import infer_environment
from .story_action import resolve_scene_action, _is_generic_action

_EMOTION_VOCAB = [
    ("scared", ["डर", "भय", "घबरा", "scared", "afraid", "fear", "terrified", "frightened"]),
    ("angry", ["गुस्सा", "क्रोध", "रोष", "angry", "furious", "rage", "mad"]),
    ("sad", ["उदास", "दुख", "दुखी", "sad", "sorrow", "grief", "unhappy", "tearful"]),
    ("surprised", ["आश्चर्य", "हैरान", "surprised", "shocked", "amazed", "astonished"]),
    ("excited", ["उत्साह", "रोमांच", "excited", "thrilled", "eager"]),
    ("sleepy", ["नींद", "सोने", "थक", "sleepy", "tired", "yawn", "drowsy"]),
    ("happy", ["खुश", "प्रसन्न", "आनंद", "मुस्कुरा", "happy", "glad", "joy", "smile", "cheerful"]),
    ("calm", ["शांत", "सुकून", "calm", "peaceful", "serene", "relaxed"]),
]

_PROP_VOCAB = [
    ("ball", ["गेंद", "ball"]),
    ("bicycle", ["साइकिल", "bicycle", "cycle"]),
    ("boat", ["नाव", "boat"]),
    ("umbrella", ["छाता", "umbrella"]),
    ("book", ["किताब", "book"]),
    ("basket", ["टोकरी", "basket"]),
    ("tree", ["पेड़", "tree"]),
    ("flower", ["फूल", "flower"]),
    ("mango", ["आम", "mango"]),
    ("sword", ["तलवार", "sword"]),
]


def _detect_emotion(text):
    raw = str(text or "")
    if not raw.strip():
        return ""
    low = raw.lower()
    for emotion, keys in _EMOTION_VOCAB:
        for k in keys:
            if k in raw or k.lower() in low:
                return emotion
    return ""


def _detect_props_in_text(text):
    raw = str(text or "")
    if not raw.strip():
        return []
    low = raw.lower()
    found = []
    for label, keys in _PROP_VOCAB:
        for k in keys:
            if k in raw or k.lower() in low:
                found.append(label)
                break
    return found


def refresh_beat_from_narration(beat, previous_beat=None):
    """Re-derive metadata from narrationText after beat subdivision."""
    text = str(beat.get("narrationText", "") or "").strip()
    if not text:
        return beat

    detected_emotion = _detect_emotion(text)
    if detected_emotion:
        beat["emotion"] = detected_emotion

    existing = str(beat.get("actionPose") or beat.get("action") or "").strip()
    if not existing or _is_generic_action(existing):
        action = resolve_scene_action(beat, previous_beat)
        beat["action"] = action
        beat["actionPose"] = action

    env = infer_environment(
        text,
        beat.get("visualPrompt", ""),
        beat.get("environment", ""),
    )
    if env:
        beat["environment"] = env
        if not str(beat.get("location") or "").strip():
            beat["location"] = env

    props = _detect_props_in_text(text)
    if props:
        beat["propsInFrame"] = props

    return beat
