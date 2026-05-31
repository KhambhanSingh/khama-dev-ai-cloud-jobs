"""Dynamic environment vocabulary (Python mirror).

Mirror of lib/videoPipeline/environment.js. Maps script location keywords to
rich background descriptions so each scene gets a distinct, script-driven
setting instead of a repeated generic background. Keep aligned with the JS side.
"""

# Hindi (Devanagari) location keyword -> background description.
HINDI_LOCATION_MAP = {
    "बारिश": "rainy scene, dark clouds, wet ground, puddles with reflections, falling rain",
    "तूफान": "violent storm, dark churning clouds, heavy rain and wind, bending trees",
    "बर्फ": "snowy landscape, white snow cover, falling snowflakes, cold pale light",
    "कोहरा": "thick foggy scene, low visibility, soft grey haze over the surroundings",
    "धूप": "bright sunny day, clear blue sky, warm golden sunlight",
    "घोंसला": "cozy bird nest of twigs high on a tree branch, leaves around, open sky",
    "गुफा": "dark rocky cave interior, stone walls, dim light from the entrance",
    "रेगिस्तान": "vast sandy desert, rolling dunes, harsh sun, distant heat haze",
    "तालाब": "calm village pond, still water with lily pads, reeds and reflections",
    "झील": "serene lake, clear reflective water, distant hills and open sky",
    "रसोई": "Indian kitchen interior, stove and utensils, warm cooking light",
    "सड़क": "open road stretching ahead, roadside trees, daytime sky",
    "पुल": "old bridge over flowing water, railings, scenic surroundings",
    "नाव": "wooden boat on rippling water, open sky, distant shore",
    "अस्पताल": "clean hospital room, white walls, bed and medical equipment, soft light",
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
    "storm": "violent storm, dark churning clouds, heavy rain and wind, bending trees",
    "snow": "snowy landscape, white snow cover, falling snowflakes, cold pale light",
    "fog": "thick foggy scene, low visibility, soft grey haze over the surroundings",
    "winter": "cold winter scene, bare trees, pale light, frosty atmosphere",
    "sunny": "bright sunny day, clear blue sky, warm golden sunlight",
    "night": "night scene, star-filled sky, moonlight, lamps, outdoor setting",
    "space": "outer space, starry cosmos, distant planets and nebula glow",
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
    "nest": "cozy bird nest of twigs high on a tree branch, leaves around, open sky",
    "cave": "dark rocky cave interior, stone walls, dim light from the entrance",
    "desert": "vast sandy desert, rolling dunes, harsh sun, distant heat haze",
    "pond": "calm village pond, still water with lily pads, reeds and reflections",
    "lake": "serene lake, clear reflective water, distant hills and open sky",
    "bridge": "old bridge over flowing water, railings, scenic surroundings",
    "boat": "wooden boat on rippling water, open sky, distant shore",
    "train": "train interior or railway platform, windows, travel atmosphere",
    "road": "open road stretching ahead, roadside trees, daytime sky",
    "hospital": "clean hospital room, white walls, bed and medical equipment, soft light",
    "office": "modern office interior, desks, computers, neutral daylight",
    "barn": "rustic farm barn, wooden beams, hay, soft natural light",
    "kitchen": "home kitchen interior, stove and utensils, warm cooking light",
    "sky": "open sky with clouds, birds in flight, soft daylight",
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
