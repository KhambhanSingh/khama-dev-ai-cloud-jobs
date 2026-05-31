"""SDXL scene images with character reference locking via img2img (single GPU pipeline)."""

import gc
import os

import numpy as np
import torch
from PIL import Image

from .logging_util import log_stage
from .validator import validate_scene_png, validate_scene_images

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

# BUG 3: Hindi/English location keywords → specific background descriptions
HINDI_LOCATION_MAP = {
    "घर":       "warm Indian home interior, cozy living room with family furniture, soft lighting",
    "स्कूल":    "Indian school classroom, green blackboard, wooden desks, bright windows",
    "बाजार":    "busy colorful Indian street market, outdoor stalls, daytime sunshine",
    "जंगल":     "dense lush green jungle, large ancient trees, dappled forest sunlight",
    "नदी":      "peaceful river bank, flowing clear water, green natural surroundings",
    "रात":      "night scene, star-filled sky, soft moonlight, peaceful outdoor setting",
    "गाँव":     "quiet Indian village, mud houses with thatched roofs, open fields",
    "राजमहल":  "ornate royal palace interior, grand arches, rich decorative walls",
    "पहाड़":    "mountain landscape, rocky terrain, fresh air, vast blue sky",
    "खेत":      "open agricultural fields, rows of crops, warm golden sunlight",
    "मंदिर":    "traditional Indian temple courtyard, stone pillars, peaceful atmosphere",
    "समुद्र":   "ocean beach, gentle waves, sandy shore, clear blue sky",
}

ENGLISH_LOCATION_MAP = {
    "forest":    "dense green forest, large trees, natural woodland setting",
    "jungle":    "dense lush jungle, tropical vegetation, dappled sunlight",
    "school":    "school classroom, blackboard, wooden desks, bright windows",
    "home":      "warm home interior, comfortable living room, family setting",
    "house":     "warm home interior, comfortable living room, family setting",
    "river":     "peaceful river bank, flowing water, green natural surroundings",
    "night":     "night scene, star-filled sky, moonlight, outdoor setting",
    "market":    "outdoor street market, colorful stalls, bustling daytime",
    "palace":    "royal palace interior, grand architecture, ornate decor",
    "mountain":  "mountain landscape, rocky terrain, vast blue sky",
    "village":   "quiet rural village, simple houses, open countryside",
    "temple":    "traditional temple courtyard, stone architecture, peaceful",
    "garden":    "lush garden, blooming flowers, green grass, sunlight",
    "field":     "open fields, golden crops, wide sky",
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


def _infer_environment(narration_text, visual_prompt, environment_field=""):
    """BUG 3: Map script narration / environment text to a specific background description."""
    hay = f"{environment_field} {narration_text} {visual_prompt}".lower()

    # Generic or missing environment — try to infer from text
    for keyword, description in HINDI_LOCATION_MAP.items():
        if keyword in hay:
            return description
    for keyword, description in ENGLISH_LOCATION_MAP.items():
        if keyword in hay:
            return description

    # Fall back to environment_field if it has useful content
    env = str(environment_field or "").strip()
    if env and env.lower() not in ("story scene", "scene", ""):
        return env

    return "natural outdoor setting with clear sky, story-appropriate surroundings"


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


def generate_reference_image(pipe, prompt, gen_w, gen_h, out_w, out_h, out_path, negative_prompt=None):
    last_err = None
    for attempt in range(1, 4):
        try:
            steps = SCENE_GEN_STEPS if attempt == 1 else SCENE_GEN_STEPS_RETRY
            image = _run_generation(
                pipe,
                prompt,
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

    # BUG 3: Resolve a specific environment description (never default to generic interior)
    environment = _infer_environment(
        beat.get("narrationText", ""),
        beat.get("visualPrompt", ""),
        beat.get("environment", ""),
    )

    # BUG 4 + BUG 10: Inject emotion-driven pose keywords
    emotion = str(beat.get("emotion", "neutral")).lower()
    pose_kw = EMOTION_POSE_KEYWORDS.get(emotion, EMOTION_POSE_KEYWORDS["neutral"])

    action = str(beat.get("action", "")).strip()

    # BUG 2: Character anchor FIRST in prompt so CLIP token attention prioritises face/outfit
    char_anchor = "; ".join(
        f"{c.get('name')}: {c.get('referencePrompt', '')[:100]}"
        for c in beat_chars[:3]
    )

    # Use the Gemini-generated English visualPrompt as the primary scene description.
    # It was written by the LLM explicitly for SDXL and is always in English.
    # DO NOT inject beat.narrationText — it is in Hindi and SDXL ignores it.
    visual_from_planner = str(beat.get("visualPrompt") or "").strip()

    if char_anchor and visual_from_planner:
        # Best case: character anchors + pre-computed English scene description + pose
        full_prompt = (
            f"{style}. "
            f"Characters: {char_anchor}. "
            f"Scene: {visual_from_planner}. "
            f"Pose: {pose_kw}."
        )
    elif char_anchor:
        # No visualPrompt: fall back to environment + action (both English from planner)
        full_prompt = (
            f"{style}. "
            f"Characters: {char_anchor}. "
            f"Background: {environment}. "
            + (f"Action: {action}. " if action else "")
            + f"Pose: {pose_kw}."
        )
    else:
        # No characters matched: use visualPrompt or environment/action
        full_prompt = (
            f"{style}. "
            f"{visual_from_planner or f'{environment}, {action}'}. "
            f"{pose_kw}."
        )

    full_prompt = _trim_prompt(full_prompt, max_words=80)

    os.makedirs(refs_dir, exist_ok=True)
    os.makedirs(scenes_dir, exist_ok=True)

    if pipe is None:
        pipe = load_img2img_model()

    init_image = None
    # BUG 2: Lower strength for more consistent character faces when ref is available
    # narrator mode → even lower (0.38) for maximum face lock
    strength_with_ref = 0.38 if narrator_mode else 0.42
    strength_with_prev = 0.45 if narrator_mode else 0.50
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
                guidance=2.0,
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
