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

DEFAULT_GEN_MAX_W = int(os.environ.get("KAGGLE_GEN_MAX_WIDTH", "1024"))
DEFAULT_GEN_MAX_H = int(os.environ.get("KAGGLE_GEN_MAX_HEIGHT", "576"))
SCENE_GEN_STEPS = int(os.environ.get("KAGGLE_SCENE_STEPS", "20"))
SCENE_GEN_STEPS_RETRY = int(os.environ.get("KAGGLE_SCENE_STEPS_RETRY", "28"))
# SDXL CLIP text encoder hard limit — longer prompts are silently truncated.
CLIP_MAX_TOKENS = int(os.environ.get("KAGGLE_CLIP_MAX_TOKENS", "77"))

# Compact negatives that fit inside the CLIP 77-token window.
SCENE_NEGATIVE_PROMPT = (
    "duplicate characters, crowd, character sheet, clones, bad anatomy, "
    "deformed hands, extra limbs, blurry, watermark, text, disney, frozen"
)

REFERENCE_NEGATIVE_PROMPT = (
    "duplicate characters, crowd, character sheet, multiple poses, sprite sheet, "
    "bad anatomy, deformed hands, blurry, watermark, text"
)

DEFAULT_NEGATIVE_PROMPT = SCENE_NEGATIVE_PROMPT

_CLIP_TOKENIZER = None

_VISUAL_PROMPT_STRIP_PHRASES = (
    "2D animated story scene, single frozen moment, sharp crisp lines.",
    "Proportional anatomy, correct hands, clear face.",
    "Original characters only, not Disney.",
    "Animated storytelling, cinematic composition, high detail, sharp focus.",
    "No text, no watermark.",
    "Exactly one character in frame.",
    "Exactly two characters interacting.",
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


def _get_clip_tokenizer(pipe=None):
    """Lazy-load CLIP tokenizer (same family SDXL uses for the 77-token limit)."""
    global _CLIP_TOKENIZER
    if _CLIP_TOKENIZER is not None:
        return _CLIP_TOKENIZER
    if pipe is not None and getattr(pipe, "tokenizer", None) is not None:
        _CLIP_TOKENIZER = pipe.tokenizer
        return _CLIP_TOKENIZER
    try:
        from transformers import CLIPTokenizer

        _CLIP_TOKENIZER = CLIPTokenizer.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            subfolder="tokenizer",
        )
    except Exception:
        _CLIP_TOKENIZER = None
    return _CLIP_TOKENIZER


def _clip_token_count(text, pipe=None):
    tok = _get_clip_tokenizer(pipe)
    if tok is None:
        return len(str(text or "").split())
    return len(tok.encode(str(text or ""), truncation=False))


def _clip_trim(text, max_tokens=None, pipe=None):
    """Trim prompt to SDXL CLIP limit so nothing important is silently dropped."""
    max_tokens = max_tokens or CLIP_MAX_TOKENS
    cleaned = " ".join(str(text or "").split()).strip()
    if not cleaned:
        return ""
    tok = _get_clip_tokenizer(pipe)
    if tok is None:
        return _trim_prompt(cleaned, max_words=max(1, max_tokens - 5))
    ids = tok.encode(cleaned, truncation=False)
    if len(ids) <= max_tokens:
        return cleaned
    return tok.decode(ids[:max_tokens], skip_special_tokens=True).strip()


def _sanitize_visual_prompt(visual):
    """Remove JS boilerplate and cap length before Python scene assembly."""
    text = " ".join(str(visual or "").split()).strip()
    for phrase in _VISUAL_PROMPT_STRIP_PHRASES:
        text = text.replace(phrase, " ")
    return " ".join(text.split())[:220]


def _short_char_label(char):
    """One tight character line for CLIP-budget prompts."""
    name = str(char.get("name", "")).strip()
    species = str(char.get("species", "")).strip()
    appearance = str(char.get("appearance", "")).strip()[:50]
    body = str(char.get("bodyShape", "")).strip()
    clothing = str(char.get("clothing", "")).strip()[:40]

    label = f"{name} the {species}" if species and species.lower() != "character" else name
    bits = [b for b in [body and f"{body} body", appearance, clothing and f"wearing {clothing}"] if b]
    if bits:
        return f"{label}, {', '.join(bits[:2])}"
    return label


def _build_compact_scene_prompt(
    style,
    pose_kw,
    environment,
    visual_from_planner,
    render_chars,
    camera_kw,
):
    """Single-layer scene prompt — no duplicate JS + Python stacking."""
    solo = "one character only" if len(render_chars) <= 1 else "two characters only"
    char_line = ", ".join(_short_char_label(c) for c in render_chars[:2])
    visual = _sanitize_visual_prompt(visual_from_planner)

    if visual and len(visual) > 30:
        if pose_kw[:24].lower() not in visual.lower():
            visual = f"{pose_kw}, {visual}"
        core = f"{style}, {visual}"
        if char_line and char_line.lower() not in visual.lower():
            core = f"{core}, {char_line}"
    else:
        env_short = str(environment or "")[:80]
        cam = str(camera_kw or "medium shot")[:30]
        core = f"{style}, {pose_kw}, {env_short}, {char_line}, {solo}, {cam}"

    if solo not in core:
        core = f"{core}, {solo}"
    return core


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
    max_prompt_words=0,
):
    steps = steps or SCENE_GEN_STEPS
    if max_prompt_words and max_prompt_words > 0:
        prompt = _trim_prompt(prompt, max_words=max_prompt_words)
    else:
        prompt = _clip_trim(prompt, pipe=pipe)
    neg = _clip_trim(negative_prompt or SCENE_NEGATIVE_PROMPT, pipe=pipe)
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
    ref_prompt = _clip_trim(
        f"{_trim_prompt(prompt, max_words=35)}. {REFERENCE_PROMPT_SUFFIX}",
        pipe=pipe,
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
                pipe=pipe,
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

    if pipe is None:
        pipe = load_img2img_model()

    beat_chars = _beat_character_entries(beat, characters)
    style = _trim_prompt(
        str(video_config.get("style", "2D cartoon")), max_words=6
    )

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

    camera_style = str(beat.get("cameraStyle", "")).strip().lower()
    camera_kw = CAMERA_KEYWORDS.get(camera_style, DEFAULT_CAMERA)
    render_chars = beat_chars[:2]
    visual_from_planner = str(beat.get("visualPrompt") or "").strip()

    raw_prompt = _build_compact_scene_prompt(
        style,
        pose_kw,
        environment,
        visual_from_planner,
        render_chars,
        camera_kw,
    )
    full_prompt = _clip_trim(raw_prompt, pipe=pipe)
    token_count = _clip_token_count(full_prompt, pipe=pipe)

    os.makedirs(refs_dir, exist_ok=True)
    os.makedirs(scenes_dir, exist_ok=True)

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
        message=(
            f"gen={gen_w}x{gen_h} out={out_w}x{out_h} tokens={token_count}/{CLIP_MAX_TOKENS} "
            f"txt2img {full_prompt[:70]}"
        ),
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
                pipe=pipe,
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
