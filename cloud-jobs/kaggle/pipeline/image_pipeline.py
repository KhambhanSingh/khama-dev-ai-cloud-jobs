"""SDXL scene images with character reference locking via img2img (single GPU pipeline)."""

import gc
import os

import numpy as np
import torch
from PIL import Image

from .logging_util import log_stage
from .validator import validate_scene_png, validate_scene_images
from .environments import infer_environment

# One img2img pipeline in VRAM (built from txt2img components once).
_IMG2IMG_PIPE = None

DEFAULT_GEN_MAX_W = int(os.environ.get("KAGGLE_GEN_MAX_WIDTH", "768"))
DEFAULT_GEN_MAX_H = int(os.environ.get("KAGGLE_GEN_MAX_HEIGHT", "432"))
SCENE_GEN_STEPS = int(os.environ.get("KAGGLE_SCENE_STEPS", "12"))
SCENE_GEN_STEPS_RETRY = int(os.environ.get("KAGGLE_SCENE_STEPS_RETRY", "16"))
SCENE_PROMPT_MAX_WORDS = int(os.environ.get("KAGGLE_SCENE_PROMPT_WORDS", "110"))
REF_PROMPT_MAX_WORDS = int(os.environ.get("KAGGLE_REF_PROMPT_WORDS", "70"))

# BUG 7: Always pass a negative prompt to prevent cloned background characters and shelf clutter
DEFAULT_NEGATIVE_PROMPT = (
    "no background people, no duplicate characters, no miniature figures, "
    "no clones, no shelf figures, crowded shelves, cluttered store interior, "
    "shop shelves with dolls, background faces, repeated character copies, "
    "multiple identical characters, character grid, sticker sheet, "
    "character model sheet, turnaround sheet, pose reference sheet, "
    "crowd of characters, many copies, tiled pattern, collage, "
    "toy store, plush toys on shelves, collection of figures, "
    "blurry, bad anatomy, deformed, distorted, watermark, text, logo"
)

REFERENCE_NEGATIVE_PROMPT = (
    DEFAULT_NEGATIVE_PROMPT
    + ", multiple poses, multiple views, character lineup, sprite sheet, "
    "contact sheet, ensemble cast, group shot, scattered figures"
)

# BUG 4 + BUG 10: Emotion → visual pose keywords injected into every scene prompt
EMOTION_POSE_KEYWORDS = {
    "happy":     "smiling broadly, arms open wide, joyful upright stance",
    "sad":       "drooped shoulders, head bowed down, tearful expression",
    "angry":     "fists clenched, intense frown, leaning forward aggressively",
    "excited":   "jumping with excitement, arms raised high, huge grin",
    "scared":    "cowering, wide frightened eyes, hunched body protectively",
    "surprised": "eyes wide open, mouth agape, stepping back in shock",
    "thinking":  "hand resting on chin, eyes looking up, thoughtful expression",
    "talking":   "one hand raised gesturing outward, mouth open expressively",
    "laughing":  "mouth wide open laughing, eyes crinkled with joy",
    "crying":    "tears running down face, sobbing, hands covering face",
    "suspense":  "tense cautious posture, looking around nervously",
    "neutral":   "natural relaxed upright pose, attentive expression",
    "calm":      "peaceful composed posture, gentle relaxed expression",
    "panic":     "running away, arms flailing, terrified expression",
    "sleepy":    "heavy drooping eyelids, slouched posture, yawning",
    "worried":   "furrowed brow, biting lip, anxious clasped hands",
    "playful":   "playful bouncy pose, mischievous grin, lively gesture",
    "emotional": "deeply moved expression, glistening eyes, hand over heart",
}

# Deterministic emotion fallback (mirrors detectEmotion in environment.js) so a
# beat that arrives "neutral" still reflects the narration's real emotion.
_EMOTION_VOCAB = [
    ("crying", ["रोया", "रोई", "रोने", "सिसक", "आँसू", "आंसू", "cry", "sob", "weep", "tears"]),
    ("laughing", ["हँस", "हंस", "ठहाक", "खिलखिला", "laugh", "giggle", "chuckle"]),
    ("scared", ["डर", "डरा", "डरी", "भयभीत", "घबरा", "सहम", "खौफ", "scared", "afraid", "fear", "terrified", "frightened"]),
    ("angry", ["गुस्सा", "गुस्से", "क्रोध", "नाराज", "क्रुद्ध", "क्रोधित", "angry", "furious", "rage", "annoyed"]),
    ("excited", ["उत्साह", "उत्साहित", "जोश", "रोमांच", "excited", "thrilled", "eager", "ecstatic"]),
    ("surprised", ["हैरान", "चौंक", "अचंभ", "आश्चर्य", "हक्का", "surprised", "shocked", "amazed", "astonished"]),
    ("worried", ["चिंता", "चिंतित", "परेशान", "फिक्र", "फ़िक्र", "बेचैन", "worried", "anxious", "nervous", "concerned"]),
    ("sad", ["उदास", "दुखी", "दुःख", "दुख", "गमगीन", "निराश", "मायूस", "sad", "unhappy", "gloomy", "grief", "sorrow"]),
    ("playful", ["शरारत", "मस्ती", "खिलवाड़", "playful", "mischiev", "frolic", "teasing"]),
    ("thinking", ["सोच", "विचार", "सोचने", "think", "ponder", "wonder", "contempl"]),
    ("suspense", ["रहस्य", "सस्पेंस", "रहस्यमय", "suspense", "tense", "mysterious", "eerie"]),
    ("sleepy", ["नींद", "सोने", "थक", "ऊँघ", "sleepy", "tired", "yawn", "drowsy"]),
    ("happy", ["खुश", "प्रसन्न", "आनंद", "मुस्कुरा", "हर्ष", "खुशी", "happy", "glad", "joy", "smile", "cheerful", "delighted"]),
    ("calm", ["शांत", "सुकून", "इत्मीनान", "calm", "peaceful", "serene", "relaxed", "tranquil"]),
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


# Script action verbs (Hindi + English) -> frozen-moment pose for SDXL.
# Mirrors lib/videoPipeline/actionPose.js.
_ACTION_VOCAB = [
    ("hugging warmly, arms wrapped around the other character, emotional embrace",
     ["गले लग", "आलिंगन", "गले मिल", "hug", "hugging", "embrace", "embracing"]),
    ("running fast, legs in full motion, body leaning forward urgently",
     ["दौड़", "भाग", "दौड़ा", "भागा", "run", "running", "chase", "chasing", "race", "escape", "flee", "sprint"]),
    ("jumping mid-air, dynamic leap, body stretched in motion",
     ["कूद", "छलांग", "उछल", "jump", "jumping", "leap", "leaping", "hop", "vault"]),
    ("walking along a path, mid-stride, natural forward movement",
     ["चल", "चला", "चली", "walk", "walking", "stroll", "wander"]),
    ("climbing upward, gripping with paws or hands, determined upward motion",
     ["चढ़", "चढ़ा", "climb", "climbing", "ascend"]),
    ("flying through the air, wings or body lifted, soaring motion",
     ["उड़", "उड़ा", "fly", "flying", "soar", "soaring", "glide"]),
    ("playing joyfully, mid-bounce or mid-game, lively playful motion",
     ["खेल", "खेला", "खेली", "play", "playing", "frolic"]),
    ("talking expressively, mouth open, one hand gesturing outward",
     ["कहा", "बोल", "बोला", "talk", "talking", "said", "says", "speak", "shout"]),
    ("looking around cautiously, head turned, alert scanning motion",
     ["देख", "देखा", "निहार", "look", "looking", "watch", "watching", "peek", "peering"]),
    ("hiding behind an object, partially concealed, peeking out nervously",
     ["छिप", "छुप", "hide", "hiding", "hidden", "conceal"]),
    ("crying with tears visible, hands near face, sorrowful moment",
     ["रोया", "रोई", "रोने", "cry", "crying", "weep", "sob", "tears"]),
    ("laughing openly, mouth wide, joyful mid-laugh expression",
     ["हँस", "हंस", "laugh", "laughing", "giggle", "chuckle"]),
    ("dancing with rhythmic motion, arms raised, celebratory movement",
     ["नाच", "नाचा", "dance", "dancing", "twirl"]),
]

_GENERIC_ACTIONS = {"", "story moment", "narration", "story scene", "neutral"}


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


def _resolve_scene_action(beat):
    """Pick the best script-driven action phrase for this beat."""
    action_pose = str(beat.get("actionPose") or "").strip()
    if action_pose and action_pose.lower() not in _GENERIC_ACTIONS:
        return action_pose

    hay = " ".join(
        str(beat.get(k) or "")
        for k in ("narrationText", "action", "summary", "visualPrompt")
    )
    detected = _detect_action_pose(hay)
    if detected:
        return detected

    action = str(beat.get("action") or "").strip()
    if action and action.lower() not in _GENERIC_ACTIONS:
        return action

    summary = str(beat.get("summary") or "").strip()
    if summary and summary.lower() not in _GENERIC_ACTIONS:
        return summary

    return "key story moment, clear visible action in frame"

# Camera framing keywords so scenes are not all the same flat wide shot.
CAMERA_KEYWORDS = {
    "wide shot":     "wide establishing shot, full scene visible",
    "wide":          "wide establishing shot, full scene visible",
    "close-up":      "close-up shot, face and expression filling the frame",
    "closeup":       "close-up shot, face and expression filling the frame",
    "medium shot":   "medium shot, character from the waist up",
    "medium":        "medium shot, character from the waist up",
    "tracking shot": "dynamic tracking shot following the movement",
    "tracking":      "dynamic tracking shot following the movement",
    "slow pan":      "slow cinematic pan across the scene",
    "pan":           "slow cinematic pan across the scene",
    "overhead":      "high overhead angle looking down on the scene",
    "low angle":     "dramatic low angle looking up at the character",
}
DEFAULT_CAMERA = "cinematic composition"

# Mood/emotion -> lighting (mirrors moodToLighting in environment.js).
MOOD_LIGHTING = {
    "sad":       "soft diffused lighting, muted tones",
    "crying":    "soft diffused lighting, muted tones",
    "happy":     "warm bright lighting, vibrant colors",
    "excited":   "warm bright lighting, vibrant colors",
    "laughing":  "warm bright lighting, vibrant colors",
    "scared":    "dramatic shadows, high contrast",
    "suspense":  "dramatic shadows, high contrast",
    "panic":     "dramatic shadows, high contrast",
    "angry":     "harsh directional lighting, intense colors",
    "calm":      "soft natural lighting, gentle tones",
}
DEFAULT_LIGHTING = "cinematic balanced lighting"

# Weather keyword -> motion / atmosphere cue for liveliness.
WEATHER_MOTION = {
    "rain":  "rain falling, wet surfaces reflecting light",
    "storm": "strong wind, swaying trees, driving rain",
    "snow":  "snowflakes drifting down softly",
    "fog":   "drifting mist, hazy air",
}


def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _round8(n):
    return max(8, int(n) // 8 * 8)


def _gen_dimensions(video_config):
    """Cap SD generation size for 15GB GPUs; output size stays in video_config."""
    out_w = int(video_config.get("width", 1280))
    out_h = int(video_config.get("height", 720))
    max_w = int(video_config.get("genMaxWidth", DEFAULT_GEN_MAX_W))
    max_h = int(video_config.get("genMaxHeight", DEFAULT_GEN_MAX_H))

    scale = min(1.0, max_w / out_w, max_h / out_h)
    gen_w = _round8(out_w * scale)
    gen_h = _round8(out_h * scale)
    return gen_w, gen_h, out_w, out_h


def _upscale_image(image, out_w, out_h):
    if image.size == (out_w, out_h):
        return image
    return image.resize((out_w, out_h), Image.Resampling.LANCZOS)


def load_img2img_model():
    """Load SDXL once: txt2img components -> img2img, drop duplicate weights."""
    global _IMG2IMG_PIPE
    if _IMG2IMG_PIPE is not None:
        return _IMG2IMG_PIPE

    from diffusers import (
        StableDiffusionXLPipeline,
        StableDiffusionXLImg2ImgPipeline,
    )

    log_stage("image", message="Loading SDXL Turbo (single pipeline)")
    base = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    _IMG2IMG_PIPE = StableDiffusionXLImg2ImgPipeline(**base.components)
    del base
    clear_gpu_memory()

    _IMG2IMG_PIPE.to("cuda")
    _IMG2IMG_PIPE.enable_attention_slicing()
    _IMG2IMG_PIPE.enable_vae_slicing()
    _IMG2IMG_PIPE.enable_vae_tiling()
    _IMG2IMG_PIPE.set_progress_bar_config(disable=True)
    return _IMG2IMG_PIPE


def _trim_prompt(prompt, max_words=55):
    return " ".join(str(prompt or "").split()[:max_words])


def _norm_name(value):
    return str(value or "").strip().casefold()


# Props/toys must never be a scene's primary character (mirrors TOY_SPECIES in
# lib/videoPipeline/characterHeuristic.js).
_PROP_SPECIES = {"teddy bear toy", "doll toy"}


def _character_priority(c):
    """main non-prop (0) < supporting non-prop (1) < prop main (2) < prop (3)."""
    is_prop = str(c.get("species", "")).strip().lower() in _PROP_SPECIES
    is_main = "main" in str(c.get("role", "")).strip().lower()
    return (2 if is_prop else 0) + (0 if is_main else 1)


def _beat_character_entries(beat, characters):
    """Resolve beat character names to registry entries (Unicode-safe).

    Sorted so real main characters come first and props/toys last, so the
    primary rendered character is never a prop.
    """
    beat_names = [_norm_name(n) for n in (beat.get("characters") or [])]
    if not beat_names:
        return []
    out = []
    seen = set()
    for c in characters or []:
        cname = _norm_name(c.get("name"))
        cid = _norm_name(c.get("id"))
        if cname in beat_names or cid in beat_names:
            if cname not in seen:
                seen.add(cname)
                out.append(c)
    out.sort(key=_character_priority)
    return out


def _safe_ref_filename(char_id):
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(char_id))
    return safe or "character"


def _char_seed(char):
    """Stable per-character seed so a character renders with a consistent
    identity across every scene (mirrors slugId-style hashing on the JS side)."""
    base = str((char or {}).get("id") or (char or {}).get("name") or "")
    h = 0
    for ch in base:
        h = (h * 31 + ord(ch)) & 0x7FFFFFFF
    return h or 1


def _make_generator(seed):
    if seed is None:
        return None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.Generator(device=device).manual_seed(int(seed) % (2 ** 31))


def _noise_init(gen_w, gen_h):
    """Colorful noise init — avoid flat grey SDXL failure mode."""
    arr = np.random.randint(0, 255, (gen_h, gen_w, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _run_generation(
    pipe,
    prompt,
    gen_w,
    gen_h,
    init_image=None,
    strength=0.58,
    steps=None,
    guidance=2.0,
    negative_prompt=None,
    seed=None,
    max_prompt_words=55,
):
    steps = steps or SCENE_GEN_STEPS
    if max_prompt_words and max_prompt_words > 0:
        prompt = _trim_prompt(prompt, max_words=max_prompt_words)
    neg = negative_prompt or DEFAULT_NEGATIVE_PROMPT
    generator = _make_generator(seed)
    with torch.inference_mode():
        if init_image is not None:
            init = init_image.convert("RGB").resize((gen_w, gen_h))
            out = pipe(
                prompt=prompt,
                negative_prompt=neg,
                image=init,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
            )
        else:
            init = _noise_init(gen_w, gen_h)
            out = pipe(
                prompt=prompt,
                negative_prompt=neg,
                image=init,
                strength=0.85,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
            )
        return out.images[0]


# Reference = ONE character portrait on a plain backdrop for identity locking in
# the text prompt. Never use "model sheet" — that triggers clone-grid output.
REFERENCE_PROMPT_SUFFIX = (
    "full body portrait, single character only, one pose, centered, "
    "plain solid white background, no other characters, no scenery, "
    "no background objects, no multiple poses"
)


def generate_reference_image(pipe, prompt, gen_w, gen_h, out_w, out_h, out_path, negative_prompt=None, seed=None):
    ref_prompt = _trim_prompt(
        f"{prompt}. {REFERENCE_PROMPT_SUFFIX}",
        max_words=REF_PROMPT_MAX_WORDS,
    )
    neg = negative_prompt or REFERENCE_NEGATIVE_PROMPT
    last_err = None
    for attempt in range(1, 4):
        try:
            steps = SCENE_GEN_STEPS if attempt == 1 else SCENE_GEN_STEPS_RETRY
            image = _run_generation(
                pipe,
                ref_prompt,
                gen_w,
                gen_h,
                init_image=None,
                strength=0.85,
                steps=steps,
                guidance=2.0,
                negative_prompt=neg,
                seed=seed,
                max_prompt_words=REF_PROMPT_MAX_WORDS,
            )
            image = _upscale_image(image, out_w, out_h)
            image.save(out_path)
            validate_scene_png(out_path)
            return out_path
        except Exception as e:
            last_err = e
            log_stage("image", message=f"ref gen attempt {attempt} failed: {e}", level="ERROR")
    raise last_err


def generate_scene_image(
    record_id,
    beat,
    characters,
    video_config,
    refs_dir,
    scenes_dir,
    previous_scene_path=None,
    pipe=None,
):
    gen_w, gen_h, out_w, out_h = _gen_dimensions(video_config)
    idx = beat.get("sceneIndex", 0)
    scene_path = os.path.join(scenes_dir, f"scene_{idx:03d}.png")

    beat_chars = _beat_character_entries(beat, characters)
    style = str(video_config.get("style", "2D kids animation, cartoon, high detail"))

    # BUG 3: Resolve a specific environment description (never default to generic interior).
    # Shared vocabulary lives in environments.py (mirrors lib/videoPipeline/environment.js).
    environment = infer_environment(
        beat.get("narrationText", ""),
        beat.get("visualPrompt", ""),
        beat.get("environment", ""),
    )

    # BUG 4 + BUG 10: Inject emotion-driven pose keywords. When the beat is
    # empty/neutral, detect a real emotion from the narration/visual prompt.
    emotion = str(beat.get("emotion", "neutral")).strip().lower()
    if not emotion or emotion == "neutral":
        detected = _detect_emotion(
            f"{beat.get('narrationText', '')} {beat.get('summary', '')} "
            f"{beat.get('action', '')} {beat.get('visualPrompt', '')}"
        )
        if detected:
            emotion = detected
    action_pose = _resolve_scene_action(beat)
    pose_kw = action_pose or EMOTION_POSE_KEYWORDS.get(
        emotion, EMOTION_POSE_KEYWORDS["neutral"]
    )

    # Camera framing (varies per scene), emotion-driven lighting, and a weather
    # motion cue so scenes feel cinematic instead of static and repetitive.
    camera_style = str(beat.get("cameraStyle", "")).strip().lower()
    camera_kw = CAMERA_KEYWORDS.get(camera_style, DEFAULT_CAMERA)
    mood = str(beat.get("mood", "")).strip().lower()
    lighting_kw = MOOD_LIGHTING.get(emotion) or MOOD_LIGHTING.get(mood) or DEFAULT_LIGHTING
    env_hay = f"{beat.get('environment', '')} {beat.get('visualPrompt', '')}".lower()
    motion_kw = ""
    for _w, _cue in WEATHER_MOTION.items():
        if _w in env_hay:
            motion_kw = _cue
            break
    cinematic_tail = (
        f"Camera: {camera_kw}. Lighting: {lighting_kw}."
        + (f" Motion: {motion_kw}." if motion_kw else "")
        + " Animated storytelling, cinematic composition, high detail."
    )

    # Render the primary character only by default. A 2nd is included solely when
    # the beat genuinely lists 2+ distinct characters (a real interaction). Drawing
    # more than this is what produced the "wall of clones" output.
    render_chars = beat_chars[:2]

    def _anchor(c):
        tags = ", ".join(
            t
            for t in (
                str(c.get("species", "")).strip()
                if str(c.get("species", "")).strip().lower() not in ("", "character")
                else "",
                str(c.get("role", "")).strip(),
            )
            if t
        )
        head = f"{c.get('name')} ({tags})" if tags else str(c.get("name"))
        colors = str(c.get("colorPalette", "")).strip()
        ref = str(c.get("referencePrompt", ""))[:130]
        extra = f" colors: {colors}." if colors else ""
        return f"{head}: {ref}{extra}"

    char_anchor = "; ".join(_anchor(c) for c in render_chars)
    solo_hint = (
        "exactly one character in frame, no crowd, no duplicates"
        if len(render_chars) <= 1
        else "exactly two characters interacting, no crowd, no duplicates"
    )

    visual_from_planner = str(beat.get("visualPrompt") or "").strip()

    # Action-first prompt: frozen story moment + setting + character identity.
    # Scenes use text-only generation (no ref/previous img2img) to avoid clone grids.
    if char_anchor and visual_from_planner:
        full_prompt = (
            f"{style}. Single story scene, one frozen moment. "
            f"Action: {pose_kw}. "
            f"Setting: {environment}. "
            f"Scene: {visual_from_planner}. "
            f"Featuring {char_anchor}. "
            f"{solo_hint}. "
            f"{cinematic_tail}"
        )
    elif char_anchor:
        full_prompt = (
            f"{style}. Single story scene, one frozen moment. "
            f"Action: {pose_kw}. "
            f"Setting: {environment}. "
            f"Featuring {char_anchor}. "
            f"{solo_hint}. "
            f"{cinematic_tail}"
        )
    else:
        full_prompt = (
            f"{style}. Single story scene, one frozen moment. "
            f"Action: {pose_kw}. "
            f"Setting: {environment}. "
            f"{visual_from_planner or 'cinematic story illustration'}. "
            f"{solo_hint}. "
            f"{cinematic_tail}"
        )

    full_prompt = _trim_prompt(full_prompt, max_words=SCENE_PROMPT_MAX_WORDS)

    os.makedirs(refs_dir, exist_ok=True)
    os.makedirs(scenes_dir, exist_ok=True)

    if pipe is None:
        pipe = load_img2img_model()

    # Ensure reference PNG exists for logging/consistency, but do NOT use it as
    # img2img init for scenes — that was cloning model-sheet grids into every frame.
    primary = beat_chars[0] if beat_chars else None
    if primary:
        cid = primary.get("id") or primary.get("name", "")
        ref_file = _safe_ref_filename(cid)
        ref_path = os.path.join(refs_dir, f"ref_{ref_file}.png")
        if not os.path.isfile(ref_path):
            log_stage("image", record_id, beat=idx, message=f"ref gen {cid}")
            try:
                generate_reference_image(
                    pipe,
                    primary.get("referencePrompt", primary.get("name", "")),
                    gen_w,
                    gen_h,
                    out_w,
                    out_h,
                    ref_path,
                    seed=_char_seed(primary),
                )
            except Exception as ref_err:
                log_stage(
                    "image",
                    record_id,
                    beat=idx,
                    message=f"ref gen skipped: {ref_err}",
                    level="ERROR",
                )

    init_image = None
    strength = 0.85

    log_stage(
        "image",
        record_id,
        beat=idx,
        message=f"gen={gen_w}x{gen_h} out={out_w}x{out_h} txt2img action={pose_kw[:40]} {full_prompt[:60]}",
    )

    scene_seed = _char_seed(primary) if primary else None

    last_err = None
    for attempt in range(1, 4):
        try:
            steps = SCENE_GEN_STEPS if attempt == 1 else SCENE_GEN_STEPS_RETRY
            image = _run_generation(
                pipe,
                full_prompt,
                gen_w,
                gen_h,
                init_image=init_image,
                strength=strength,
                steps=steps,
                guidance=3.0,
                seed=scene_seed,
                max_prompt_words=SCENE_PROMPT_MAX_WORDS,
            )
            image = _upscale_image(image, out_w, out_h)
            image.save(scene_path)
            validate_scene_png(scene_path)
            clear_gpu_memory()
            return scene_path
        except Exception as e:
            last_err = e
            log_stage(
                "image",
                record_id,
                beat=idx,
                message=f"attempt {attempt}/3 failed: {e}",
                level="ERROR",
            )
            strength = min(0.75, strength + 0.08)

    clear_gpu_memory()
    raise last_err or RuntimeError(f"scene generation failed for beat {idx}")


def generate_all_scenes(record_id, beats, characters, video_config, work_dirs):
    refs_dir = os.path.join(work_dirs["root"], "refs")
    scenes_dir = os.path.join(work_dirs["root"], "scenes")
    os.makedirs(refs_dir, exist_ok=True)
    os.makedirs(scenes_dir, exist_ok=True)

    if video_config.get("videoStyle"):
        video_config = {**video_config, "style": video_config.get("videoStyle")}

    gen_w, gen_h, out_w, out_h = _gen_dimensions(video_config)
    pipe = load_img2img_model()

    # BUG 2: Pre-generate reference images for ALL characters before scene loop.
    # This ensures every character has a locked visual anchor from the first scene.
    for char in characters or []:
        cid = char.get("id") or char.get("name", "")
        ref_file = _safe_ref_filename(cid)
        ref_path = os.path.join(refs_dir, f"ref_{ref_file}.png")
        if not os.path.isfile(ref_path):
            log_stage("image", record_id, message=f"pre-gen ref for character: {cid}")
            try:
                generate_reference_image(
                    pipe,
                    char.get("referencePrompt", char.get("name", "")),
                    gen_w,
                    gen_h,
                    out_w,
                    out_h,
                    ref_path,
                    seed=_char_seed(char),
                )
            except Exception as ref_err:
                log_stage(
                    "image",
                    record_id,
                    message=f"ref pre-gen failed for {cid}: {ref_err}",
                    level="ERROR",
                )

    paths = []
    for beat in beats:
        path = generate_scene_image(
            record_id,
            beat,
            characters,
            video_config,
            refs_dir,
            scenes_dir,
            previous_scene_path=None,
            pipe=pipe,
        )
        paths.append(path)

    validate_scene_images(paths)
    log_stage("image", record_id, message=f"images_done count={len(paths)}")
    return paths
