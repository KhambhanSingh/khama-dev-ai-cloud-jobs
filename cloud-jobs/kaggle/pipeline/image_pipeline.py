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
SCENE_GEN_STEPS = int(os.environ.get("KAGGLE_SCENE_STEPS", "6"))
SCENE_GEN_STEPS_RETRY = int(os.environ.get("KAGGLE_SCENE_STEPS_RETRY", "10"))

# BUG 7: Always pass a negative prompt to prevent cloned background characters and shelf clutter
DEFAULT_NEGATIVE_PROMPT = (
    "no background people, no duplicate characters, no miniature figures, "
    "no clones, no shelf figures, crowded shelves, cluttered store interior, "
    "shop shelves with dolls, background faces, repeated character copies, "
    "multiple identical characters, character grid, sticker sheet, "
    "toy store, plush toys on shelves, collection of figures, "
    "blurry, bad anatomy, deformed, distorted, watermark, text, logo"
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
}

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


def _beat_character_entries(beat, characters):
    """Resolve beat character names to registry entries (Unicode-safe)."""
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
    return out


def _safe_ref_filename(char_id):
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(char_id))
    return safe or "character"


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
):
    steps = steps or SCENE_GEN_STEPS
    prompt = _trim_prompt(prompt)
    neg = negative_prompt or DEFAULT_NEGATIVE_PROMPT
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
            )
        return out.images[0]


# Reference anchors must be the character ALONE on a clean backdrop, otherwise
# SDXL invents a setting (e.g. toy-store shelves) that then bleeds into every
# scene through img2img. This suffix is applied to reference prompts only.
REFERENCE_PROMPT_SUFFIX = (
    "full body, single character, centered, plain solid neutral background, "
    "character model sheet, no other characters, no scenery, no background objects"
)


def generate_reference_image(pipe, prompt, gen_w, gen_h, out_w, out_h, out_path, negative_prompt=None):
    ref_prompt = _trim_prompt(f"{prompt}. {REFERENCE_PROMPT_SUFFIX}", max_words=80)
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
                negative_prompt=negative_prompt,
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
    narrator_mode = bool(video_config.get("narratorMode"))
    style = str(video_config.get("style", "2D kids animation, cartoon, high detail"))

    # BUG 3: Resolve a specific environment description (never default to generic interior).
    # Shared vocabulary lives in environments.py (mirrors lib/videoPipeline/environment.js).
    environment = infer_environment(
        beat.get("narrationText", ""),
        beat.get("visualPrompt", ""),
        beat.get("environment", ""),
    )

    # BUG 4 + BUG 10: Inject emotion-driven pose keywords
    emotion = str(beat.get("emotion", "neutral")).lower()
    pose_kw = EMOTION_POSE_KEYWORDS.get(emotion, EMOTION_POSE_KEYWORDS["neutral"])

    action = str(beat.get("action", "")).strip()

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
    cinematic_tail = f"Camera: {camera_kw}. Lighting: {lighting_kw}." + (
        f" Motion: {motion_kw}." if motion_kw else ""
    )

    # Render the primary character only by default. A 2nd is included solely when
    # the beat genuinely lists 2+ distinct characters (a real interaction). Drawing
    # more than this is what produced the "wall of clones" output.
    render_chars = beat_chars[:2]
    char_anchor = "; ".join(
        f"{c.get('name')}: {c.get('referencePrompt', '')[:90]}"
        for c in render_chars
    )
    solo_hint = (
        "exactly one character in frame"
        if len(render_chars) <= 1
        else "exactly two characters interacting"
    )

    # Use the Gemini-generated English visualPrompt as the primary scene description.
    # It was written by the LLM explicitly for SDXL and is always in English.
    # DO NOT inject beat.narrationText — it is in Hindi and SDXL ignores it.
    visual_from_planner = str(beat.get("visualPrompt") or "").strip()

    if char_anchor and visual_from_planner:
        # Best case: lead with the story SETTING so the background follows the
        # script, then add the character anchor, pose, camera and lighting.
        full_prompt = (
            f"{style}. "
            f"Scene: {visual_from_planner}. "
            f"Featuring {char_anchor}. "
            f"{solo_hint}. Pose: {pose_kw}. "
            f"{cinematic_tail}"
        )
    elif char_anchor:
        # No visualPrompt: lead with environment + action (both English from planner)
        full_prompt = (
            f"{style}. "
            f"Setting: {environment}. "
            + (f"{action}. " if action else "")
            + f"Featuring {char_anchor}. "
            + f"{solo_hint}. Pose: {pose_kw}. "
            + f"{cinematic_tail}"
        )
    else:
        # No characters matched: use visualPrompt or environment/action
        full_prompt = (
            f"{style}. "
            f"{visual_from_planner or f'{environment}, {action}'}. "
            f"{pose_kw}. {cinematic_tail}"
        )

    full_prompt = _trim_prompt(full_prompt, max_words=90)

    os.makedirs(refs_dir, exist_ok=True)
    os.makedirs(scenes_dir, exist_ok=True)

    if pipe is None:
        pipe = load_img2img_model()

    init_image = None
    # Strength must be high enough that the SCENE background follows the script
    # instead of cloning the reference image's backdrop. Too low (0.42) baked the
    # reference's setting into every scene. img2img still preserves identity here.
    strength_with_ref = 0.50 if narrator_mode else 0.62
    strength_with_prev = 0.55 if narrator_mode else 0.60
    strength_no_ref = 0.85  # full generation when no reference exists

    strength = strength_no_ref

    ref_path = None
    primary = beat_chars[0] if beat_chars else None
    if primary:
        cid = primary.get("id") or primary.get("name", "")
        ref_file = _safe_ref_filename(cid)
        ref_path = os.path.join(refs_dir, f"ref_{ref_file}.png")
        if not os.path.isfile(ref_path):
            log_stage("image", record_id, beat=idx, message=f"ref gen {cid}")
            generate_reference_image(
                pipe,
                primary.get("referencePrompt", primary.get("name", "")),
                gen_w,
                gen_h,
                out_w,
                out_h,
                ref_path,
            )

    if ref_path and os.path.isfile(ref_path):
        try:
            validate_scene_png(ref_path)
            init_image = Image.open(ref_path).convert("RGB")
            strength = strength_with_ref
        except (ValueError, OSError):
            init_image = None
            ref_path = None

    if init_image is None and previous_scene_path and os.path.isfile(previous_scene_path):
        try:
            validate_scene_png(previous_scene_path)
            init_image = Image.open(previous_scene_path).convert("RGB")
            strength = strength_with_prev
        except (ValueError, OSError):
            init_image = None

    log_stage(
        "image",
        record_id,
        beat=idx,
        message=f"gen={gen_w}x{gen_h} out={out_w}x{out_h} strength={strength:.2f} {full_prompt[:80]}",
    )

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
                )
            except Exception as ref_err:
                log_stage(
                    "image",
                    record_id,
                    message=f"ref pre-gen failed for {cid}: {ref_err}",
                    level="ERROR",
                )

    paths = []
    prev = None
    for beat in beats:
        path = generate_scene_image(
            record_id,
            beat,
            characters,
            video_config,
            refs_dir,
            scenes_dir,
            previous_scene_path=prev,
            pipe=pipe,
        )
        paths.append(path)
        prev = path

    validate_scene_images(paths)
    log_stage("image", record_id, message=f"images_done count={len(paths)}")
    return paths
