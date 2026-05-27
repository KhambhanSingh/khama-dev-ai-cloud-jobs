
"""
🎬 Smart Video Generator v2.1 - FIXED
"""

import subprocess, sys, os, json, time, glob, gc, shutil
import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from PIL import Image
from gtts import gTTS
from pydub import AudioSegment
from pydub.effects import speedup

# -------------------------
# AUTO INSTALL
# -------------------------
def _ensure_packages():
    needed = {
        "gtts": "gTTS",
        "pydub": "pydub",
        "moviepy": "moviepy"
    }

    for module, pkg in needed.items():
        try:
            __import__(module)
        except:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", pkg]
            )

_ensure_packages()

# -------------------------
# GPU
# -------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

torch.backends.cuda.matmul.allow_tf32 = True

# -------------------------
# EMOTIONS
# -------------------------
EMOTION_MAP = {
    "neutral": ("standing calmly", 0.20),
    "happy": ("smiling brightly", 0.25),
    "sad": ("looking down", 0.20),
    "excited": ("jumping happily", 0.30),
    "angry": ("frowning", 0.25),
    "thinking": ("thinking", 0.20),
}

# -------------------------
# DIRS
# -------------------------
def job_work_dirs(record_id):
    rid = str(record_id)

    root = os.path.join("cloud-jobs", "work", rid)

    return {
        "root": root,
        "frames": os.path.join(root, "frames"),
        "audio": os.path.join(root, "audio"),
        "videos": os.path.join(root, "videos"),
    }

def ensure_job_dirs(d):
    for k in d.values():
        os.makedirs(k, exist_ok=True)

# -------------------------
# MEMORY
# -------------------------
def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()

# -------------------------
# MODEL LOAD
# -------------------------
_BASE_PIPE = None
_IMG2IMG_PIPE = None

def load_base_model():
    global _BASE_PIPE

    if _BASE_PIPE:
        return _BASE_PIPE

    print("🔧 Loading SDXL Turbo")

    _BASE_PIPE = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
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

    print("🔧 Loading Img2Img")

    _IMG2IMG_PIPE = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    ).to("cuda")

    _IMG2IMG_PIPE.enable_attention_slicing()
    _IMG2IMG_PIPE.enable_vae_slicing()
    _IMG2IMG_PIPE.enable_vae_tiling()
    _IMG2IMG_PIPE.set_progress_bar_config(disable=True)

    return _IMG2IMG_PIPE

# -------------------------
# PROMPT FIX
# -------------------------
def trim_prompt(prompt, max_words=18):
    return " ".join(prompt.split()[:max_words])

def build_prompt(characters, emotion):
    action, _ = EMOTION_MAP.get(emotion, ("standing", 0.20))

    short_chars = []

    for c in characters[:3]:
        short_chars.append(c[:20])

    joined = ", ".join(short_chars)

    p = (
        f"{joined}, "
        f"{action}, "
        f"cartoon style, colorful, high quality"
    )

    return trim_prompt(p)

# -------------------------
# AUDIO
# -------------------------
def generate_audio(record_id, text, lang="hi"):
    d = job_work_dirs(record_id)

    ensure_job_dirs(d)

    mp3 = os.path.join(d["audio"], f"{record_id}.mp3")
    wav = os.path.join(d["audio"], f"{record_id}.wav")

    tts = gTTS(text=text, lang=lang, slow=False)

    tts.save(mp3)

    audio = AudioSegment.from_file(mp3)

    audio.export(wav, format="wav")

    duration = len(audio) / 1000

    return wav, duration

# -------------------------
# FRAMES
# -------------------------
def generate_frames(
    record_id,
    characters,
    emotion_timeline,
    audio_duration
):
    d = job_work_dirs(record_id)

    ensure_job_dirs(d)

    base_pipe = load_base_model()

    frames_dir = d["frames"]

    FPS = 6

    total_frames = max(
        int(audio_duration * FPS),
        len(emotion_timeline) * 3
    )

    current = 0

    image_size = 512

    frames_per_emotion = max(
        1,
        total_frames // len(emotion_timeline)
    )

    for emotion in emotion_timeline:

        n = frames_per_emotion

        prompt = build_prompt(
            characters,
            emotion
        )

        print(
            f"🎭 {emotion} | prompt={prompt}"
        )

        for _ in range(n):

            if current >= total_frames:
                break

            frame_path = os.path.join(
                frames_dir,
                f"{record_id}_{current:04d}.png"
            )

            try:

                with torch.inference_mode():

                    out = base_pipe(
                        prompt=prompt,
                        height=image_size,
                        width=image_size,
                        num_inference_steps=1,
                        guidance_scale=0.0
                    )

                    image = out.images[0]

                image.save(frame_path)

                current += 1

                if current % 5 == 0:
                    print(
                        f"Frame {current}/{total_frames}"
                    )

                if current % 10 == 0:
                    clear_gpu_memory()

            except Exception as e:

                print(
                    f"❌ Frame {current}: {e}"
                )

                continue

    clear_gpu_memory()

    return current

# -------------------------
# VIDEO
# -------------------------
def create_video_from_frames(
    record_id,
    audio_path,
    total_frames,
    audio_duration
):
    d = job_work_dirs(record_id)

    video_path = os.path.join(
        d["videos"],
        f"{record_id}.mp4"
    )

    fps = max(
        8,
        min(total_frames / audio_duration, 24)
    )

    frame_pattern = os.path.join(
        d["frames"],
        f"{record_id}_%04d.png"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        frame_pattern,
        "-i",
        audio_path,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        video_path,
    ]

    subprocess.run(cmd, check=True)

    return video_path

# -------------------------
# JOB
# -------------------------
def process_job(job_data):

    record_id = job_data["recordId"]

    characters = job_data["characters"]

    emotions = job_data["emotions"]

    narration = job_data["narration"]

    lang = job_data.get("language", "hi")

    print(f"\n🎬 START {record_id}")

    audio_path, duration = generate_audio(
        record_id,
        narration,
        lang
    )

    total_frames = generate_frames(
        record_id,
        characters,
        emotions,
        duration
    )

    video = create_video_from_frames(
        record_id,
        audio_path,
        total_frames,
        duration
    )

    print(f"✅ DONE {video}")

    return {
        "recordId": record_id,
        "video": video,
        "audio": audio_path,
    }

# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":

    print("🚀 Ready")

    if torch.cuda.is_available():
        print(
            torch.cuda.get_device_name(0)
        )