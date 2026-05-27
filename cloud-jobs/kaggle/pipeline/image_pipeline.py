"""SDXL scene images with character reference locking via img2img."""

import gc
import os

import torch
from PIL import Image

from .logging_util import log_stage

_BASE_PIPE = None
_IMG2IMG_PIPE = None


def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_base_model():
    global _BASE_PIPE
    if _BASE_PIPE:
        return _BASE_PIPE

    from diffusers import StableDiffusionXLPipeline

    log_stage("image", message="Loading SDXL Turbo txt2img")
    _BASE_PIPE = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    ).to("cuda")
    _BASE_PIPE.enable_attention_slicing()
    _BASE_PIPE.enable_vae_slicing()
    _BASE_PIPE.enable_vae_tiling()
    _BASE_PIPE.set_progress_bar_config(disable=True)
    return _BASE_PIPE


def load_img2img_model():
    global _IMG2IMG_PIPE
    if _IMG2IMG_PIPE:
        return _IMG2IMG_PIPE

    from diffusers import StableDiffusionXLImg2ImgPipeline

    log_stage("image", message="Loading SDXL Turbo img2img")
    _IMG2IMG_PIPE = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    ).to("cuda")
    _IMG2IMG_PIPE.enable_attention_slicing()
    _IMG2IMG_PIPE.enable_vae_slicing()
    _IMG2IMG_PIPE.enable_vae_tiling()
    _IMG2IMG_PIPE.set_progress_bar_config(disable=True)
    return _IMG2IMG_PIPE


def _trim_prompt(prompt, max_words=55):
    return " ".join(str(prompt or "").split()[:max_words])


def generate_reference_image(pipe, prompt, width, height, out_path):
    prompt = _trim_prompt(prompt)
    with torch.inference_mode():
        out = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=2,
            guidance_scale=0.0,
        )
        image = out.images[0]
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
):
    width = int(video_config.get("width", 1280))
    height = int(video_config.get("height", 720))
    idx = beat.get("sceneIndex", 0)
    scene_path = os.path.join(scenes_dir, f"scene_{idx:03d}.png")

    visual = beat.get("visualPrompt") or ""
    if not visual:
        char_bits = ", ".join(
            c.get("referencePrompt", c.get("name", ""))[:80]
            for c in (characters or [])[:3]
        )
        visual = f"{beat.get('environment', 'scene')}, {char_bits}, {beat.get('action', '')}"

    visual = _trim_prompt(visual)
    style = str(video_config.get("style", "2D kids animation, cartoon, high detail"))
    full_prompt = f"{style}, {visual}"

    os.makedirs(refs_dir, exist_ok=True)
    os.makedirs(scenes_dir, exist_ok=True)

    base_pipe = load_base_model()
    img_pipe = load_img2img_model()

    init_image = None
    strength = 0.58

    # Pick reference: first character in beat, else previous scene
    beat_names = [str(n).lower() for n in (beat.get("characters") or [])]
    ref_path = None
    for c in characters or []:
        cid = c.get("id") or c.get("name", "")
        cname = str(c.get("name", "")).lower()
        cid_l = str(cid).lower()
        if beat_names and cname not in beat_names and cid_l not in beat_names:
            continue
        ref_path = os.path.join(refs_dir, f"ref_{cid_l}.png")
        if not os.path.isfile(ref_path):
            log_stage("image", record_id, beat=idx, message=f"ref gen {cid}")
            generate_reference_image(
                base_pipe,
                c.get("referencePrompt", c.get("name", "")),
                width,
                height,
                ref_path,
            )
        break

    if ref_path and os.path.isfile(ref_path):
        init_image = Image.open(ref_path).convert("RGB").resize((width, height))
    elif previous_scene_path and os.path.isfile(previous_scene_path):
        init_image = Image.open(previous_scene_path).convert("RGB").resize((width, height))
        strength = 0.52

    log_stage("image", record_id, beat=idx, message=full_prompt[:120])

    try:
        with torch.inference_mode():
            if init_image is not None:
                out = img_pipe(
                    prompt=full_prompt,
                    image=init_image,
                    strength=strength,
                    num_inference_steps=2,
                    guidance_scale=0.0,
                )
            else:
                out = base_pipe(
                    prompt=full_prompt,
                    height=height,
                    width=width,
                    num_inference_steps=2,
                    guidance_scale=0.0,
                )
            image = out.images[0]
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
        )
        paths.append(path)
        prev = path

    return paths
