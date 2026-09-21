"""SDXL prompt sanitization — English-only, forbidden clone/crowd vocabulary."""

import re

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")

FORBIDDEN_PROMPT_PHRASES = (
    "character sheet",
    "reference sheet",
    "model sheet",
    "sprite sheet",
    "multiple poses",
    "pose variations",
    "turnaround sheet",
    "front side back",
    "lineup",
    "character lineup",
    "group shot",
    "group of people",
    "many characters",
    "many rabbits",
    "rabbit family",
    "family group",
    "duplicate characters",
)

POSITIVE_ONLY_WORDS = ("crowd", "clones", "clone")

_GENERIC_ACTION_MARKERS = (
    "story moment",
    "key story moment",
    "clear frozen story moment",
    "story scene",
)

_NEGATION_PROTECT_RE = re.compile(
    r"\bno\s+(crowd|clones?|duplicate\s+\w+|extra\s+people|character\s+sheet)\b",
    re.I,
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
    """Strip forbidden vocabulary; preserve 'no crowd' / 'no clones' negation phrases."""
    out = str(text or "")
    protected = []

    def _protect(m):
        key = f"__NEG{len(protected)}__"
        protected.append((key, m.group(0)))
        return key

    out = _NEGATION_PROTECT_RE.sub(_protect, out)

    for phrase in FORBIDDEN_PROMPT_PHRASES:
        out = re.sub(re.escape(phrase), " ", out, flags=re.I)

    for word in POSITIVE_ONLY_WORDS:
        out = re.sub(rf"(?<!no\s)\b{re.escape(word)}\b", " ", out, flags=re.I)

    for key, original in protected:
        out = out.replace(key, original)

    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r",\s*,", ",", out).strip().strip(",")
    return out


def pick_english_story_event(beat):
    """Story event from action/scriptEvent fields only — not environment-only visualPrompt."""
    for key in ("scriptEvent", "action", "actionPose", "summary"):
        val = strip_forbidden_prompt_words(str(beat.get(key) or "").strip())
        if val and is_english_prompt_text(val) and not _is_generic_action(val):
            return val
    return ""


def pick_english_beat_line(beat, fields=None):
    """English production fields — never narrationText."""
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
    appearance = str(char.get("appearance") or "").strip()[:45]
    if appearance:
        bits.append(appearance)
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
    """Compact CLIP-safe portrait prompt (~35 words). Anti-clone is in negative prompt."""
    desc = describe_character_for_prompt(char, 55)
    style = str(video_style or "2D cartoon").strip()
    return strip_forbidden_prompt_words(
        f"{style}, ONE character full body portrait, white background, front view, centered, {desc}"
    )


def validate_sdxl_prompt(prompt):
    issues = []
    text = str(prompt or "").strip()
    if not text:
        issues.append("empty prompt")
    if contains_devanagari(text):
        issues.append("contains Hindi/Devanagari")
    lower = text.lower()
    for phrase in FORBIDDEN_PROMPT_PHRASES:
        if phrase.lower() in lower:
            issues.append(f"forbidden: {phrase}")
    return issues
