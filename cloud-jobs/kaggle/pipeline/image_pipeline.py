"""SDXL scene images with character reference locking via img2img (single GPU pipeline)."""

import gc
import os

import torch
from PIL import Image

from .logging_util import log_stage

# One img2img pipeline in VRAM (built from txt2img components once).
_IMG2IMG_PIPE = None

DEFAULT_GEN_MAX_W = int(os.environ.get("KAGGLE_GEN_MAX_WIDTH", "768"))
DEFAULT_GEN_MAX_H = int(os.environ.get("KAGGLE_GEN_MAX_HEIGHT", "432"))


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


def _neutral_init(gen_w, gen_h):
    return Image.new("RGB", (gen_w, gen_h), (128, 128, 128))


def _run_generation(pipe, prompt, gen_w, gen_h, init_image=None, strength=0.58):
    prompt = _trim_prompt(prompt)
    with torch.inference_mode():
        if init_image is not None:
            init = init_image.convert("RGB").resize((gen_w, gen_h))
            out = pipe(
                prompt=prompt,
                image=init,
                strength=strength,
                num_inference_steps=2,
                guidance_scale=0.0,
            )
        else:
            # No reference: gray init + high strength ≈ txt2img
            out = pipe(
                prompt=prompt,
                image=_neutral_init(gen_w, gen_h),
                strength=0.92,
                num_inference_steps=2,
                guidance_scale=0.0,
            )
        return out.images[0]


def generate_reference_image(pipe, prompt, gen_w, gen_h, out_w, out_h, out_path):
    image = _run_generation(pipe, prompt, gen_w, gen_h, init_image=None)
    image = _upscale_image(image, out_w, out_h)
    image.save(out_path)
    return out_path


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
    visual = beat.get("visualPrompt") or ""
    if not visual and beat_chars:
        char_bits = "; ".join(
            f"{c.get('name')}: {c.get('referencePrompt', '')[:120]}"
            for c in beat_chars[:6]
        )
        visual = (
            f"Characters: {char_bits}. "
            f"Environment: {beat.get('environment', 'scene')}. "
            f"Action: {beat.get('action', '')}."
        )
    elif not visual:
        visual = f"{beat.get('environment', 'scene')}, {beat.get('action', '')}"

    style = str(video_config.get("style", "2D kids animation, cartoon, high detail"))
    full_prompt = f"{style}, {visual}"
    if beat_chars:
        for c in beat_chars:
            name = str(c.get("name") or "")
            if name and name not in full_prompt:
                full_prompt += f" Include {name}."
    full_prompt = _trim_prompt(full_prompt, max_words=80)

    os.makedirs(refs_dir, exist_ok=True)
    os.makedirs(scenes_dir, exist_ok=True)

    if pipe is None:
        pipe = load_img2img_model()

    init_image = None
    strength = 0.58

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
        init_image = Image.open(ref_path).convert("RGB")
    elif previous_scene_path and os.path.isfile(previous_scene_path):
        init_image = Image.open(previous_scene_path).convert("RGB")
        strength = 0.52

    log_stage(
        "image",
        record_id,
        beat=idx,
        message=f"gen={gen_w}x{gen_h} out={out_w}x{out_h} {full_prompt[:80]}",
    )

    try:
        image = _run_generation(
            pipe, full_prompt, gen_w, gen_h, init_image=init_image, strength=strength
        )
        image = _upscale_image(image, out_w, out_h)
        image.save(scene_path)
    except Exception as e:
        log_stage("image", record_id, beat=idx, message=f"fail: {e}", level="ERROR")
        if previous_scene_path and os.path.isfile(previous_scene_path):
            import shutil

            shutil.copy2(previous_scene_path, scene_path)
        elif ref_path and os.path.isfile(ref_path):
            import shutil

            shutil.copy2(ref_path, scene_path)
        else:
            raise

    clear_gpu_memory()
    return scene_path


def generate_all_scenes(record_id, beats, characters, video_config, work_dirs):
    refs_dir = os.path.join(work_dirs["root"], "refs")
    scenes_dir = os.path.join(work_dirs["root"], "scenes")
    os.makedirs(scenes_dir, exist_ok=True)

    if video_config.get("videoStyle"):
        video_config = {**video_config, "style": video_config.get("videoStyle")}

    pipe = load_img2img_model()
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

    return paths
