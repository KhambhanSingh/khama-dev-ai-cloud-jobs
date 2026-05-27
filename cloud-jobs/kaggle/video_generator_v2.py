"""
YouTube cartoon full_video generator — imported by git_queue_processor.py only.
Do not run this file directly on Kaggle; use git_queue_processor.py instead.

Fixes:
- huggingface-hub version conflict fixed (downgrade to 0.24.0)
- CUDA OOM: एक time पर एक model
- edge-tts: +0% format fix
- VAE float32 fix
- CLIP 77 token limit fix
- Kernel restart workaround
"""

import subprocess, sys, os

# ML pins handled by git_queue_processor via kaggle_deps before this module imports.

# =========================================================
# MAIN IMPORTS
# =========================================================
import gc, shutil, json, time, math, asyncio
import torch
from PIL import Image, ImageDraw, ImageFont
from pydub import AudioSegment
from pydub.effects import normalize
import edge_tts

# =========================================================
# HF LOGIN
# =========================================================
def hf_login(token):
    if not token:
        print("⚠️  HF_TOKEN नहीं है — downloads slow हो सकते हैं")
        return
    try:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        os.environ["HF_TOKEN"] = token
        print("✅ HuggingFace login successful!")
    except Exception as e:
        print(f"⚠️  HF login: {e}")

# =========================================================
# GPU CONFIG
# =========================================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = True

# =========================================================
# CONSTANTS
# =========================================================
VIDEO_W         = 1280
VIDEO_H         = 720
GEN_W           = 704
GEN_H           = 400
FPS             = 8
INFERENCE_STEPS = 2
GUIDANCE_SCALE  = 0.0
IMG2IMG_STR     = 0.35
SCENE_HOLD_SEC  = 3.5

# =========================================================
# VOICE MAP
# =========================================================
VOICE_MAP = {
    "hi":      "hi-IN-SwaraNeural",
    "hi_male": "hi-IN-MadhurNeural",
    "en":      "en-IN-NeerjaNeural",
}

EMOTION_VOICE = {
    "happy":    {"rate": "+10%", "pitch": "+5Hz",  "volume": "+10%"},
    "excited":  {"rate": "+20%", "pitch": "+10Hz", "volume": "+15%"},
    "sad":      {"rate": "-15%", "pitch": "-5Hz",  "volume": "+0%"},
    "angry":    {"rate": "+15%", "pitch": "+8Hz",  "volume": "+20%"},
    "scared":   {"rate": "+10%", "pitch": "+6Hz",  "volume": "+0%"},
    "neutral":  {"rate": "+0%",  "pitch": "+0Hz",  "volume": "+0%"},
    "thinking": {"rate": "-10%", "pitch": "+0Hz",  "volume": "+0%"},
    "proud":    {"rate": "+0%",  "pitch": "+3Hz",  "volume": "+5%"},
    "laughing": {"rate": "+15%", "pitch": "+10Hz", "volume": "+10%"},
    "crying":   {"rate": "-20%", "pitch": "-8Hz",  "volume": "+0%"},
    "shocked":  {"rate": "+5%",  "pitch": "+5Hz",  "volume": "+10%"},
    "running":  {"rate": "+15%", "pitch": "+5Hz",  "volume": "+10%"},
    "walking":  {"rate": "+0%",  "pitch": "+0Hz",  "volume": "+0%"},
}

EMOTION_MAP = {
    "neutral":  "standing calmly, gentle smile, relaxed pose",
    "happy":    "smiling brightly, arms wide open, joyful",
    "sad":      "looking down, drooping shoulders, tearful",
    "excited":  "jumping up, arms raised, thrilled",
    "angry":    "frowning, fists clenched, intense",
    "thinking": "hand on chin, looking up, contemplating",
    "running":  "running fast, legs in motion, dynamic",
    "shocked":  "eyes wide, hands on cheeks, mouth open",
    "proud":    "standing tall, chest forward, confident",
    "scared":   "cowering, wide eyes, hands raised",
    "laughing": "laughing heartily, head back, joyful",
    "crying":   "crying, tears on face, face in hands",
    "walking":  "walking steadily, calm expression",
}

BACKGROUND_MAP = {
    "forest":   "enchanted forest, golden sunlight, lush green trees",
    "village":  "Indian village, thatched houses, wheat fields",
    "palace":   "royal palace, golden pillars, red carpets",
    "night":    "starry night sky, glowing full moon",
    "river":    "river bank, crystal water, green trees",
    "school":   "classroom, chalkboard, wooden desks",
    "market":   "colorful Indian market, shops, festive",
    "mountain": "snow mountains, rocky peaks, white clouds",
    "desert":   "golden sand dunes, blue sky, palm trees",
    "home":     "cozy Indian home, warm lighting, colorful",
    "default":  "colorful landscape, flowers, sunny blue sky",
}

# =========================================================
# DIRS
# =========================================================
def job_work_dirs(record_id):
    root = os.path.join("cloud-jobs", "work", str(record_id))
    return {
        "root":    root,
        "frames":  os.path.join(root, "frames"),
        "audio":   os.path.join(root, "audio"),
        "videos":  os.path.join(root, "videos"),
        "results": os.path.join(root, "results"),
    }

def ensure_job_dirs(d):
    for v in d.values():
        os.makedirs(v, exist_ok=True)

# =========================================================
# GPU MEMORY
# =========================================================
def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# =========================================================
# MODEL — एक time पर एक
# =========================================================
_PIPE = None

def unload_model():
    global _PIPE
    if _PIPE is not None:
        del _PIPE
        _PIPE = None
        clear_gpu()
        print("   🗑️  Model unloaded")

def load_text2img():
    global _PIPE
    unload_model()
    print("🔧 Text2Img loading...")
    from diffusers import StableDiffusionXLPipeline
    _PIPE = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    ).to("cuda")
    _PIPE.enable_attention_slicing(1)
    _PIPE.vae.enable_slicing()
    _PIPE.vae.enable_tiling()
    _PIPE.vae.to(torch.float32)
    _PIPE.set_progress_bar_config(disable=True)
    print("✅ Text2Img ready!")
    return _PIPE

def load_img2img():
    global _PIPE
    unload_model()
    print("🔧 Img2Img loading...")
    from diffusers import StableDiffusionXLImg2ImgPipeline
    _PIPE = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    ).to("cuda")
    _PIPE.enable_attention_slicing(1)
    _PIPE.vae.enable_slicing()
    _PIPE.vae.enable_tiling()
    _PIPE.vae.to(torch.float32)
    _PIPE.set_progress_bar_config(disable=True)
    print("✅ Img2Img ready!")
    return _PIPE

# =========================================================
# PROMPT — 77 tokens के अंदर
# =========================================================
def build_prompt(characters, emotion, background):
    action = EMOTION_MAP.get(emotion, "standing calmly")
    bg     = BACKGROUND_MAP.get(background, BACKGROUND_MAP["default"])
    chars  = " and ".join([c.strip()[:40] for c in characters[:2]])
    return (
        f"{chars}, {action}, "
        f"{bg}, "
        f"2D cartoon style, vibrant colors, "
        f"expressive faces, children storybook art"
    )

def build_negative():
    return "realistic, photo, 3D, ugly, blurry, watermark, extra limbs"

# =========================================================
# FONT + SUBTITLE
# =========================================================
def get_font(size):
    paths = [
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in paths:
        if os.path.exists(p):
            try: return ImageFont.truetype(p, size)
            except: pass
    return ImageFont.load_default()

def add_subtitle(frame, text):
    if not text or not text.strip():
        return frame
    img  = frame.copy().convert("RGB")
    W, H = img.size
    font = get_font(max(28, W // 42))
    words, lines, cur = text.split(), [], ""
    for w in words:
        test = (cur + " " + w).strip()
        if len(test) > 42:
            if cur: lines.append(cur.strip())
            cur = w
        else:
            cur = test
    if cur: lines.append(cur.strip())
    line_h = int(font.size * 1.4)
    pad    = 12
    box_h  = len(lines) * line_h + pad * 2
    box_y  = H - box_h - 30
    bar    = Image.new("RGBA", (W, box_h), (0, 0, 0, 155))
    base   = img.convert("RGBA")
    base.paste(bar, (0, box_y), bar)
    img    = base.convert("RGB")
    draw   = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        y = box_y + pad + i * line_h
        draw.text((W//2+2, y+2), line, font=font, fill=(0,0,0),     anchor="mt")
        draw.text((W//2,   y),   line, font=font, fill=(255,255,230), anchor="mt")
    return img

def add_watermark(frame, name):
    if not name: return frame
    img  = frame.copy()
    W, _ = img.size
    draw = ImageDraw.Draw(img)
    font = get_font(16)
    draw.text((W-8, 8), f"▶ {name}", font=font,
              fill=(255,255,200), anchor="rt",
              stroke_width=1, stroke_fill=(0,0,0))
    return img

# =========================================================
# IMAGE GENERATION
# =========================================================
def generate_scene_images(record_id, scenes, characters, channel_name):
    all_scene_images = []
    prev_image       = None

    print(f"🖼️  {len(scenes)} scenes के लिए images...")

    for si, scene in enumerate(scenes):
        emotion    = scene.get("emotion", "neutral")
        background = scene.get("background", "default")
        prompt     = build_prompt(characters, emotion, background)
        negative   = build_negative()
        scene_imgs = []

        print(f"   Scene {si+1}/{len(scenes)}: {emotion}")

        if prev_image is None:
            # पहला scene: text2img
            pipe = load_text2img()
            gen  = torch.Generator("cuda").manual_seed(42)
            with torch.inference_mode():
                out = pipe(
                    prompt=prompt, negative_prompt=negative,
                    height=GEN_H, width=GEN_W,
                    num_inference_steps=INFERENCE_STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                    generator=gen,
                )
            img = out.images[0].resize((VIDEO_W, VIDEO_H), Image.LANCZOS)
            scene_imgs.append(img)
            prev_image = img
            unload_model()
        else:
            # बाकी scenes: img2img
            pipe = load_img2img()
            ref  = prev_image.resize((GEN_W, GEN_H), Image.LANCZOS)
            gen  = torch.Generator("cuda").manual_seed(42 + si)
            with torch.inference_mode():
                out = pipe(
                    prompt=prompt, negative_prompt=negative,
                    image=ref, strength=IMG2IMG_STR,
                    num_inference_steps=max(2, INFERENCE_STEPS),
                    guidance_scale=GUIDANCE_SCALE,
                    generator=gen,
                )
            img = out.images[0].resize((VIDEO_W, VIDEO_H), Image.LANCZOS)
            scene_imgs.append(img)
            prev_image = img
            unload_model()

        # Variation image (subtle motion feel)
        pipe = load_img2img()
        ref2 = prev_image.resize((GEN_W, GEN_H), Image.LANCZOS)
        gen2 = torch.Generator("cuda").manual_seed(200 + si)
        with torch.inference_mode():
            out2 = pipe(
                prompt=prompt, negative_prompt=negative,
                image=ref2, strength=0.20,
                num_inference_steps=2,
                guidance_scale=GUIDANCE_SCALE,
                generator=gen2,
            )
        scene_imgs.append(out2.images[0].resize((VIDEO_W, VIDEO_H), Image.LANCZOS))
        unload_model()

        all_scene_images.append(scene_imgs)
        print(f"      ✅ done")

    return all_scene_images

# =========================================================
# FRAME WRITING
# =========================================================
def write_frames(record_id, scenes, all_scene_images, scene_durations, channel_name):
    d = job_work_dirs(record_id)
    ensure_job_dirs(d)
    all_paths  = []
    global_idx = 0
    for si, (scene, imgs) in enumerate(zip(scenes, all_scene_images)):
        subtitle = scene.get("subtitle", "")
        duration = scene_durations[si] if si < len(scene_durations) else SCENE_HOLD_SEC
        n_frames = max(int(duration * FPS), FPS * 2)
        for fi in range(n_frames):
            frame = add_subtitle(imgs[fi % len(imgs)], subtitle)
            frame = add_watermark(frame, channel_name)
            path  = os.path.join(d["frames"], f"f{global_idx:06d}.png")
            frame.save(path)
            all_paths.append(path)
            global_idx += 1
    print(f"✅ {len(all_paths)} frames written\n")
    return all_paths

# =========================================================
# AUDIO
# =========================================================
async def _tts(text, voice, rate, pitch, volume, out):
    await edge_tts.Communicate(
        text=text, voice=voice,
        rate=rate, pitch=pitch, volume=volume,
    ).save(out)

def generate_audio(record_id, scenes, lang="hi"):
    d = job_work_dirs(record_id)
    ensure_job_dirs(d)
    voice    = VOICE_MAP.get(lang, VOICE_MAP["hi"])
    segments = []
    print(f"🎙️ Audio ({voice})...")
    for i, scene in enumerate(scenes):
        text    = scene.get("subtitle", "")
        emotion = scene.get("emotion", "neutral")
        ev      = EMOTION_VOICE.get(emotion, EMOTION_VOICE["neutral"])
        mp3     = os.path.join(d["audio"], f"s{i:03d}.mp3")
        if not text.strip():
            segments.append(AudioSegment.silent(duration=int(SCENE_HOLD_SEC*1000)))
            continue
        try:
            asyncio.get_event_loop().run_until_complete(
                _tts(text, voice, ev["rate"], ev["pitch"], ev["volume"], mp3)
            )
            seg = normalize(AudioSegment.from_file(mp3))
            min_ms = int(SCENE_HOLD_SEC * 1000)
            if len(seg) < min_ms:
                seg += AudioSegment.silent(duration=min_ms - len(seg))
            segments.append(seg)
            print(f"   Scene {i+1}: {len(seg)/1000:.1f}s | {emotion}")
        except Exception as e:
            print(f"   ⚠️  Scene {i+1}: {e}")
            segments.append(AudioSegment.silent(duration=int(SCENE_HOLD_SEC*1000)))

    combined = segments[0]
    for seg in segments[1:]:
        combined += seg

    bg = os.path.join("cloud-jobs", "assets", "bg_music.mp3")
    if os.path.exists(bg):
        try:
            bgm = AudioSegment.from_file(bg) - 18
            while len(bgm) < len(combined): bgm += bgm
            combined = combined.overlay(bgm[:len(combined)])
            print("   🎵 BG music mixed!")
        except Exception as e:
            print(f"   ⚠️  BG skip: {e}")

    wav = os.path.join(d["audio"], f"{record_id}.wav")
    combined.export(wav, format="wav")
    durs  = [len(s)/1000.0 for s in segments]
    total = len(combined)/1000.0
    print(f"✅ Audio: {total:.1f}s\n")
    return wav, total, durs

# =========================================================
# VIDEO ASSEMBLY
# =========================================================
def create_video(record_id, audio_path, frame_paths):
    d          = job_work_dirs(record_id)
    video_path = os.path.join(d["videos"], f"{record_id}.mp4")
    list_path  = os.path.join(d["root"], "frames.txt")

    with open(list_path, "w") as f:
        for fp in frame_paths:
            f.write(f"file '{os.path.abspath(fp)}'\n")
            f.write(f"duration {1.0/FPS:.6f}\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", list_path,
        "-i", audio_path,
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-r", str(FPS),
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
        "-shortest", "-movflags", "+faststart",
        video_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"FFmpeg: {r.stderr[-400:]}")
    if not os.path.exists(video_path):
        raise RuntimeError("Video file created नहीं हुई")

    size_mb = os.path.getsize(video_path) / (1024*1024)
    print(f"✅ Video: {size_mb:.1f}MB → {video_path}\n")
    return video_path

def cleanup_frames(record_id):
    d = job_work_dirs(record_id)
    deleted = 0
    for f in os.listdir(d["frames"]):
        if f.endswith(".png"):
            try: os.remove(os.path.join(d["frames"], f)); deleted += 1
            except: pass
    if deleted: print(f"🗑️  {deleted} frames cleaned")

# =========================================================
# MAIN
# =========================================================
def process_job(job_data, hf_token=None):
    record_id    = job_data["recordId"]
    lang         = job_data.get("language", "hi")
    characters   = job_data.get("characters", ["cartoon character"])
    channel_name = job_data.get("channelName", "")
    scenes       = job_data.get("scenes", [])

    if hf_token:
        hf_login(hf_token)

    # Fallback: scenes नहीं → emotions से
    if not scenes:
        emotions  = job_data.get("emotions", ["neutral"])
        narration = job_data.get("narration", "")
        words     = narration.split()
        wpb       = max(1, len(words) // max(1, len(emotions)))
        scenes    = [
            {
                "emotion":    em,
                "background": "default",
                "subtitle":   " ".join(words[i*wpb:(i+1)*wpb]),
            }
            for i, em in enumerate(emotions)
        ]

    d = job_work_dirs(record_id)
    ensure_job_dirs(d)

    print(f"\n{'='*60}")
    print(f"🎬 Job: {record_id}")
    print(f"   Scenes: {len(scenes)} | Lang: {lang}")
    print(f"{'='*60}\n")

    try:
        start = time.time()

        print("[1/3] 🎙️ Audio...")
        audio_path, total_dur, scene_durs = generate_audio(record_id, scenes, lang)

        print("[2/3] 🖼️ Images...")
        all_imgs = generate_scene_images(record_id, scenes, characters, channel_name)

        print("[2b] Writing frames...")
        frame_paths = write_frames(record_id, scenes, all_imgs, scene_durs, channel_name)

        print("[3/3] 🎬 Video...")
        video_path = create_video(record_id, audio_path, frame_paths)
        cleanup_frames(record_id)

        t = round(time.time() - start, 2)
        result = {
            "recordId": record_id, "status": "SUCCESS",
            "video": video_path, "audio": audio_path,
            "processingTime": t,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        result_path = os.path.join(d["results"], f"{record_id}.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"{'='*60}\n✅ DONE! Time: {t}s\n{'='*60}\n")
        return result

    except Exception as e:
        import traceback; err = traceback.format_exc()
        print(f"❌ FAILED: {e}\n{err}")
        with open(os.path.join(d["results"], f"{record_id}_FAILED.json"), "w") as f:
            json.dump({"recordId": record_id, "status": "FAILED",
                       "error": str(e), "traceback": err,
                       "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
                      f, indent=2)
        unload_model(); clear_gpu(); raise

if __name__ == "__main__":
    print("🚀 Video Generator v4.2")
    if torch.cuda.is_available():
        print(f"🎮 {torch.cuda.get_device_name(0)}")
        print(f"💾 {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB VRAM")