"""Dynamic environment vocabulary (Python mirror).

Mirror of lib/videoPipeline/environment.js. Maps script location keywords to
rich background descriptions so each scene gets a distinct, script-driven
setting instead of a repeated generic background. Keep aligned with the JS side.
"""

# Hindi (Devanagari) location keyword -> background description.
HINDI_LOCATION_MAP = {
    "घर": "warm Indian home interior, cozy living room with family furniture, soft lighting",
    "कमरा": "indoor room with furniture, window light, lived-in details",
    "स्कूल": "Indian school classroom, green blackboard, wooden desks, bright windows",
    "बाजार": "busy colorful Indian street market, outdoor stalls, signs, daytime sunshine",
    "बाज़ार": "busy colorful Indian street market, outdoor stalls, signs, daytime sunshine",
    "जंगल": "dense lush green jungle, large ancient trees, leaves, dappled forest sunlight",
    "नदी": "peaceful river bank, flowing clear water with reflections, green surroundings",
    "रात": "night scene, star-filled sky, soft moonlight, street lamps, peaceful outdoor setting",
    "गाँव": "quiet Indian village, mud houses with thatched roofs, trees, road, cattle, open fields",
    "गांव": "quiet Indian village, mud houses with thatched roofs, trees, road, cattle, open fields",
    "राजमहल": "ornate royal palace interior, grand arches, rich decorative walls",
    "महल": "ornate royal palace interior, grand arches, rich decorative walls",
    "पहाड़": "mountain landscape, rocky terrain, fresh air, vast blue sky",
    "खेत": "open agricultural fields, rows of crops, warm golden sunlight",
    "मंदिर": "traditional Indian temple, spiritual stone architecture, pillars, peaceful atmosphere",
    "समुद्र": "ocean beach, gentle waves, sandy shore, clear blue sky",
    "बारिश": "rainy scene, dark clouds, wet ground, puddles with reflections, falling rain",
    "शहर": "city street with buildings, traffic, signs and city lights",
    "बगीचा": "lush garden park, blooming flowers, green grass, benches, trees, sunlight",
    "पार्क": "public park with trees, grass, benches and walking paths",
    "त्योहार": "festival scene, colorful decorations, string lights, celebration atmosphere",
    "युद्ध": "dramatic battlefield, action environment, dust and tension",
}

# English location keyword -> background description.
# Atmospheric / genre / weather keywords are listed first so they take
# priority over generic structural words (e.g. "horror house" -> horror).
ENGLISH_LOCATION_MAP = {
    "horror": "dark eerie setting, heavy shadows, unsettling mood, dim light",
    "fantasy": "magical fantasy world, glowing elements, otherworldly scenery",
    "battle": "dramatic battlefield, action environment, dust and tension",
    "festival": "festival scene, colorful decorations, string lights, celebration mood",
    "rain": "rainy scene, dark clouds, wet ground, puddles with reflections, falling rain",
    "night": "night scene, star-filled sky, moonlight, lamps, outdoor setting",
    "forest": "dense green forest, large trees, leaves, natural woodland setting",
    "jungle": "dense lush jungle, tropical vegetation, dappled sunlight",
    "classroom": "school classroom, blackboard, wooden desks, bright windows",
    "school": "school classroom, blackboard, wooden desks, bright windows",
    "river": "peaceful river bank, flowing water with reflections, green surroundings",
    "market": "outdoor street market, colorful stalls, signs, bustling daytime crowd",
    "palace": "royal palace interior, grand architecture, ornate decor",
    "temple": "traditional temple, spiritual stone architecture, peaceful courtyard",
    "mountain": "mountain landscape, rocky terrain, vast blue sky",
    "village": "quiet rural village, mud houses, trees, road, cattle, open countryside",
    "city": "city street with tall buildings, traffic, signs and bright lights",
    "garden": "lush garden, blooming flowers, green grass, sunlight",
    "park": "public park with trees, grass and benches",
    "beach": "ocean beach, gentle waves, sandy shore, clear blue sky",
    "ocean": "open ocean with waves, horizon and blue sky",
    "field": "open fields, golden crops, wide sky",
    "home": "warm home interior, comfortable living room, family setting, soft light",
    "house": "warm home interior, comfortable living room, family setting, soft light",
    "room": "indoor room with furniture and window light",
}

_GENERIC_ENV = ("story scene", "scene", "background", "")


def infer_environment(narration_text, visual_prompt, environment_field=""):
    """Map script narration / environment text to a specific background description.

    Tries the Hindi and English vocabularies first, then falls back to any
    non-generic environment text, and finally a safe default.
    """
    raw = f"{environment_field} {narration_text} {visual_prompt}"
    hay = raw.lower()

    for keyword, description in HINDI_LOCATION_MAP.items():
        if keyword in raw:
            return description
    for keyword, description in ENGLISH_LOCATION_MAP.items():
        if keyword in hay:
            return description

    env = str(environment_field or "").strip()
    if env and env.lower() not in _GENERIC_ENV:
        return env

    return "natural outdoor setting with clear sky, story-appropriate surroundings"
