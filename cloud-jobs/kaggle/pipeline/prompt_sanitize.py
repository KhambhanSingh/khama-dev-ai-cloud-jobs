"""SDXL prompt sanitization — English-only, forbidden clone/crowd vocabulary."""

import re

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")

FORBIDDEN_PROMPT_WORDS = (
    "character sheet",
    "reference sheet",
    "model sheet",
    "sprite sheet",
    "multiple poses",
    "pose variations",
    "turnaround",
    "front side back",
    "lineup",
    "character lineup",
    "group shot",
    "group of",
    "crowd",
    "many characters",
    "many rabbits",
    "rabbit family",
    "family group",
    "duplicate characters",
    "clone",
    "clones",
)

_GENERIC_ACTION_MARKERS = (
    "story moment",
    "key story moment",
    "clear frozen story moment",
    "story scene",
)


def contains_devanagari(text):
    return bool(DEVANAGARI_RE.search(str(text or "")))


def is_english_prompt_text(text):
    t = str(text or "").strip()
    if not t or len(t) < 3:
        return False
    if contains_devanagari(t):
        return False
    return bool(re.search(r"[a-zA-Z]", t))


def _is_generic_action(value):
    v = str(value or "").strip().lower()
    if not v:
        return True
    return any(m in v for m in _GENERIC_ACTION_MARKERS)


def strip_forbidden_prompt_words(text):
    out = str(text or "")
    for word in FORBIDDEN_PROMPT_WORDS:
        out = re.sub(re.escape(word), " ", out, flags=re.I)
    return re.sub(r"\s{2,}", " ", out).strip()


def pick_english_beat_line(beat, fields=None):
    """English production fields only — never narrationText."""
    if fields is None:
        fields = (
            "action",
            "actionPose",
            "scriptEvent",
            "summary",
            "visualPrompt",
            "beatTitle",
        )
    for key in fields:
        val = strip_forbidden_prompt_words(str(beat.get(key) or "").strip())
        if val and is_english_prompt_text(val) and not _is_generic_action(val):
            return val
    return ""


def describe_character_for_prompt(char, max_len=90):
    name = str(char.get("name") or "").strip()
    if not name:
        return ""
    species = str(char.get("species") or "").strip()
    bits = []
    if species and species.lower() != "character":
        bits.append(species)
    age = str(char.get("age") or "").strip()
    if age:
        bits.append(f"age {age}")
    body = str(char.get("bodyShape") or "").strip()
    if body:
        bits.append(f"{body} body")
    appearance = str(char.get("appearance") or "").strip()[:50]
    if appearance:
        bits.append(appearance)
    eyes = str(char.get("eyes") or "").strip()
    if eyes:
        bits.append(f"{eyes} eyes")
    hair = str(char.get("hairstyle") or "").strip()
    if hair:
        bits.append(f"{hair} hair")
    clothing = str(char.get("clothing") or "").strip()
    if clothing:
        bits.append(f"wearing {clothing}")
    desc = ", ".join(bits) if bits else "story character"
    return f"{name}, {desc}"[:max_len]


def format_scene_character_labels(chars, max_len=180):
    labels = []
    for i, c in enumerate((chars or [])[:2]):
        letter = chr(65 + i)
        desc = describe_character_for_prompt(c, max(max_len // 2 - 12, 40))
        if desc:
            labels.append(f"Character {letter}: {desc}")
    return ". ".join(labels)[:max_len]


def build_reference_portrait_prompt(char, video_style="2D cartoon"):
    desc = describe_character_for_prompt(char, 100)
    style = str(video_style or "2D cartoon").strip()
    return strip_forbidden_prompt_words(
        "Full body portrait of ONE character standing alone, centered composition, "
        "plain solid white background, single pose, front-facing, entire body visible, "
        f"{desc}, {style} style, "
        "no duplicate characters, no crowd, no extra people, no character sheet, "
        "no turnaround sheet, no multiple poses"
    )


def validate_sdxl_prompt(prompt):
    issues = []
    text = str(prompt or "").strip()
    if not text:
        issues.append("empty prompt")
    if contains_devanagari(text):
        issues.append("contains Hindi/Devanagari")
    lower = text.lower()
    for word in FORBIDDEN_PROMPT_WORDS:
        if word.lower() in lower:
            issues.append(f"forbidden: {word}")
    return issues
