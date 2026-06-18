"""Context-aware story action extraction (mirrors lib/videoPipeline/actionPose.js)."""

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
}

_GENERIC_POSE_PREFIXES = (
    "standing proudly",
    "walking along a path",
    "looking around cautiously",
    "sitting and listening",
    "talking expressively",
)


def _is_generic_action(value):
    v = str(value or "").strip().lower()
    if not v:
        return True
    if v in _GENERIC_ACTIONS:
        return True
    if re.match(r"^(standing|sitting|walking|looking|character scene|story scene)\b", v):
        return True
    return any(v.startswith(p) for p in _GENERIC_POSE_PREFIXES)


def _story_event_rules():
    """Return (match_fn, action_fn) pairs — context before keywords."""

    def waiting_cave(t, _b, _e, env):
        at_cave = re.search(r"(गुफा|cave)", t, re.I)
        loc = "outside a cave" if at_cave else (f"at {env[:40]}" if env else "outside")
        return f"waiting patiently {loc}, carefully watching the entrance"

    rules = [
        (
            lambda t, *_: re.search(r"(इंतजार|प्रतीक्ष|wait(ing)?|awaiting)", t, re.I)
            and re.search(r"(गुफा|cave|प्रवेश|entrance|द्वार|door|बाहर|outside)", t, re.I),
            waiting_cave,
        ),
        (
            lambda t, *_: re.search(r"(इंतजार|प्रतीक्ष|wait(ing)?|awaiting)", t, re.I),
            lambda *_: "waiting patiently, alert posture, eyes fixed on the path ahead",
        ),
        (
            lambda t, *_: re.search(r"(छिप|छुप|hide|hiding|hidden)", t, re.I)
            and re.search(r"(डर|भय|डरते|frighten|scared|afraid|terrified)", t, re.I),
            lambda t, *_: (
                "hiding behind bushes with a frightened expression, body crouched low"
                if re.search(r"(झाड़|bush|shrub|ped|tree|rock|पत्थर|दीवार|wall)", t, re.I)
                else "hiding behind cover with a frightened expression, body crouched low"
            ),
        ),
        (
            lambda t, *_: re.search(r"(छिप|छुप|hide|hiding|hidden|conceal)", t, re.I),
            lambda *_: "hiding behind an object, partially concealed, peeking out nervously",
        ),
        (
            lambda t, *_: re.search(r"(तोड़|tod|pick(ing)?|pluck(ing)?|तोड़ रह)", t, re.I)
            and re.search(r"(आम|mango|fruit|फल|branch|शाख)", t, re.I),
            lambda *_: "reaching toward a mango branch, hand grasping ripe fruit mid-pick",
        ),
        (
            lambda t, *_: re.search(r"(तलवार|sword|weapon|हथियार|knife|चाकू)", t, re.I)
            and re.search(r"(निकाल|draw|unsheathe|pull(ing)? out|brandish)", t, re.I),
            lambda t, *_: (
                "drawing a sword and advancing toward a monster with weapon raised"
                if re.search(r"(राक्षस|monster|enemy|शत्रु|demon|dragon)", t, re.I)
                else "drawing a sword and stepping forward boldly"
            ),
        ),
        (
            lambda t, *_: re.search(r"(बढ़|advance|approach|charging|charge|मार्च)", t, re.I)
            and re.search(r"(राक्षस|monster|enemy|शत्रु|opponent)", t, re.I),
            lambda *_: "advancing toward an enemy with weapon ready, determined battle stance",
        ),
        (
            lambda t, *_: re.search(r"(खींच|pull(ing)?|drag(ging)?|बचा|rescue|save)", t, re.I)
            and re.search(r"(नदी|river|water|पानी|stream|talab|pond)", t, re.I),
            lambda *_: "pulling a friend out of the river, urgent rescue gesture",
        ),
        (
            lambda t, *_: re.search(r"(खोल|open(ing)?|unlock)", t, re.I)
            and re.search(r"(संदुक|chest|treasure|खजान|box|trunk)", t, re.I),
            lambda *_: "carefully opening an ancient treasure chest, hands on the lid",
        ),
        (
            lambda t, *_: re.search(r"(खिड़की|window|jharokha)", t, re.I)
            and re.search(r"(देख|look|gaze|peek|outside|बाहर)", t, re.I),
            lambda *_: "reaching toward a moonlit window and looking outside",
        ),
        (
            lambda t, *_: re.search(r"(गले लग|आलिंगन|hug|embrace|embracing)", t, re.I),
            lambda *_: "hugging warmly, arms wrapped around the other character, emotional embrace",
        ),
        (
            lambda t, *_: re.search(r"(दौड़|भाग|run|running|chase|escape|flee|sprint)", t, re.I)
            and re.search(r"(जंगल|forest|path|road|mud|danger|खतर)", t, re.I),
            lambda t, _b, _e, env: (
                "running fast through a muddy forest path while escaping danger"
                if re.search(r"(जंगल|forest)", t, re.I)
                else f"running fast through {env[:35]} while escaping danger" if env else "running fast while escaping danger"
            ),
        ),
        (
            lambda t, *_: re.search(r"(दौड़|भाग|run|running|chase|escape|flee|sprint)", t, re.I),
            lambda *_: "running fast, legs in full motion, body leaning forward urgently",
        ),
        (
            lambda t, *_: re.search(r"(कूद|jump|leap|छलांग)", t, re.I),
            lambda *_: "jumping mid-air, dynamic leap, body stretched in motion",
        ),
        (
            lambda t, *_: re.search(r"(रोया|रोई|cry|crying|weep|sob|tears)", t, re.I),
            lambda *_: "crying with tears visible, hands near face, sorrowful moment",
        ),
        (
            lambda t, *_: re.search(r"(हँस|हंस|laugh|laughing|giggle)", t, re.I),
            lambda *_: "laughing openly, mouth wide, joyful mid-laugh expression",
        ),
        (
            lambda t, *_: re.search(r"(लड़|fight|fighting|battle|attack|strike|combat)", t, re.I),
            lambda *_: "fighting dynamically, combat pose with weapon or claws raised",
        ),
    ]
    return rules


_KEYWORD_VOCAB = [
    (
        "hugging warmly, arms wrapped around the other character, emotional embrace",
        ["गले लग", "आलिंगन", "hug", "hugging", "embrace"],
    ),
    (
        "running fast, legs in full motion, body leaning forward urgently",
        ["दौड़", "भाग", "run", "running", "chase", "sprint"],
    ),
    (
        "jumping mid-air, dynamic leap, body stretched in motion",
        ["कूद", "छलांग", "jump", "jumping", "leap"],
    ),
    (
        "climbing upward, gripping with paws or hands, determined upward motion",
        ["चढ़", "climb", "climbing"],
    ),
    (
        "flying through the air, wings or body lifted, soaring motion",
        ["उड़", "fly", "flying", "soar"],
    ),
]


def _keyword_fallback(text):
    raw = str(text or "")
    if not raw.strip():
        return ""
    low = raw.lower()
    for pose, keys in _KEYWORD_VOCAB:
        for k in keys:
            if k in raw or k.lower() in low:
                return pose
    return ""


def extract_story_action(beat, previous_beat=None):
    """Context-aware action from narration + beat fields."""
    narration = str(beat.get("narrationText") or "").strip()
    summary = str(beat.get("scriptEvent") or beat.get("summary") or "").strip()
    emotion = str(beat.get("emotion") or beat.get("mood") or "").strip()
    environment = str(beat.get("environment") or beat.get("location") or "").strip()
    prev_bit = ""
    if previous_beat:
        prev_bit = str(previous_beat.get("summary") or previous_beat.get("narrationText") or "")[:80]

    hay = " ".join(
        x for x in [narration, summary, beat.get("beatTitle"), prev_bit] if x
    ).strip()
    if not hay:
        return ""

    for match_fn, action_fn in _story_event_rules():
        if match_fn(hay, beat, emotion, environment):
            action = action_fn(hay, beat, emotion, environment)
            if action and not _is_generic_action(action):
                return action

    if re.search(r"(कहा|बोल|said|speak|shout|ask|reply)", hay, re.I):
        em = f", {emotion} expression" if emotion and emotion != "neutral" else ""
        return f"talking expressively, mouth open, one hand gesturing outward{em}"

    return _keyword_fallback(hay)


def resolve_scene_action(beat, previous_beat=None):
    """Story-aware action resolution — trust payload first, keywords last."""
    for key in ("actionPose", "action", "scriptEvent", "summary"):
        val = str(beat.get(key) or "").strip()
        if val and not _is_generic_action(val):
            return val

    from_context = extract_story_action(beat, previous_beat)
    if from_context:
        return from_context

    title = str(beat.get("beatTitle") or "").strip()
    if title and not _is_generic_action(title):
        return title

    return "key story moment, clear visible action in frame"
