"""Refresh beat action/emotion/environment from a narration slice (no torch)."""

from .environments import infer_environment

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

_ACTION_VOCAB = [
    ("hugging warmly, arms wrapped around the other character, emotional embrace",
     ["गले लग", "आलिंगन", "hug", "hugging", "embrace"]),
    ("running fast, legs in full motion, body leaning forward urgently",
     ["दौड़", "भाग", "run", "running", "chase", "sprint"]),
    ("jumping mid-air, dynamic leap, body stretched in motion",
     ["कूद", "छलांग", "jump", "jumping", "leap"]),
    ("walking along a path, mid-stride, natural forward movement",
     ["चल", "चला", "walk", "walking", "stroll"]),
    ("climbing upward, gripping with paws or hands, determined upward motion",
     ["चढ़", "climb", "climbing"]),
    ("flying through the air, wings or body lifted, soaring motion",
     ["उड़", "fly", "flying", "soar"]),
    ("playing joyfully, mid-bounce or mid-game, lively playful motion",
     ["खेल", "play", "playing"]),
    ("talking expressively, mouth open, one hand gesturing outward",
     ["कहा", "बोल", "talk", "talking", "said", "speak", "shout"]),
    ("looking around cautiously, head turned, alert scanning motion",
     ["देख", "look", "looking", "watch", "watching"]),
    ("hiding behind an object, partially concealed, peeking out nervously",
     ["छिप", "छुप", "hide", "hiding"]),
    ("crying with tears visible, hands near face, sorrowful moment",
     ["रोया", "रोई", "cry", "crying", "weep", "tears"]),
    ("laughing openly, mouth wide, joyful mid-laugh expression",
     ["हँस", "हंस", "laugh", "laughing"]),
    ("dancing with rhythmic motion, arms raised, celebratory movement",
     ["नाच", "dance", "dancing"]),
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
]

_GENERIC_ACTIONS = {"", "story moment", "narration", "story scene", "neutral"}


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


def _detect_action_pose(text):
    raw = str(text or "")
    if not raw.strip():
        return ""
    low = raw.lower()
    for pose, keys in _ACTION_VOCAB:
        for k in keys:
            if k in raw or k.lower() in low:
                return pose
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


def refresh_beat_from_narration(beat):
    """Re-derive metadata from narrationText after beat subdivision."""
    text = str(beat.get("narrationText", "") or "").strip()
    if not text:
        return beat

    detected_emotion = _detect_emotion(text)
    if detected_emotion:
        beat["emotion"] = detected_emotion

    detected_action = _detect_action_pose(text)
    if detected_action:
        beat["action"] = detected_action
        beat["actionPose"] = detected_action
    else:
        action = str(beat.get("action") or "").strip()
        if not action or action.lower() in _GENERIC_ACTIONS:
            beat["action"] = "key story moment, clear visible action in frame"
            beat["actionPose"] = beat["action"]

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
