
"""
Smart Video Generator v2 — legacy emotion frames + v2 beat pipeline.
"""

import subprocess
import sys
import os
import gc
import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from PIL import Image
from gtts import gTTS
from pydub import AudioSegment

# -------------------------
# AUTO INSTALL
# -------------------------
def _ensure_packages():
    needed = {
        "gtts": "gTTS",
        "pydub": "pydub",
    }
    for module, pkg in needed.items():
        try:
            __import__(module)
        except ImportError:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", pkg]
            )


_ensure_packages()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = True

EMOTION_MAP = {
    "neutral": ("standing calmly", 0.20),
    "happy": ("smiling brightly", 0.25),
    "sad": ("looking down", 0.20),
    "excited": ("jumping happily", 0.30),
    "angry": ("frowning", 0.25),
    "thinking": ("thinking", 0.20),
    "proud": ("standing proud", 0.22),
    "running": ("running actively", 0.28),
    "relaxed": ("relaxed pose", 0.20),
    "sleepy": ("sleepy eyes", 0.18),
    "asleep": ("sleeping peacefully", 0.18),
    "shocked": ("shocked expression", 0.25),
    "panic": ("panicking", 0.30),
    "scared": ("scared cowering", 0.25),
    "laughing": ("laughing joyfully", 0.28),
    "crying": ("crying sadly", 0.22),
    "calm": ("calm standing", 0.20),
}


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


def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()


_BASE_PIPE = None


def load_base_model():
    global _BASE_PIPE
    if _BASE_PIPE:
        return _BASE_PIPE
    print("Loading SDXL Turbo (legacy)")
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


def trim_prompt(prompt, max_words=18):
    return " ".join(prompt.split()[:max_words])


def build_prompt(characters, emotion):
    action, _ = EMOTION_MAP.get(emotion, ("standing", 0.20))
    short_chars = []
    for c in characters[:3]:
        short_chars.append(str(c)[:120] if isinstance(c, str) else str(c.get("referencePrompt", c))[:120])
    joined = ", ".join(short_chars)
    p = f"{joined}, {action}, cartoon style, colorful, high quality"
    return trim_prompt(p, 55)


def generate_audio(record_id, text, lang="hi"):
    d = job_work_dirs(record_id)
    ensure_job_dirs(d)
    mp3 = os.path.join(d["audio"], f"{record_id}.mp3")
    wav = os.path.join(d["audio"], f"{record_id}.wav")
    gTTS(text=text, lang=lang, slow=False).save(mp3)
    audio = AudioSegment.from_file(mp3)
    audio.export(wav, format="wav")
    return wav, len(audio) / 1000


def generate_frames(record_id, characters, emotion_timeline, audio_duration, image_size=512):
    d = job_work_dirs(record_id)
    ensure_job_dirs(d)
    base_pipe = load_base_model()
    frames_dir = d["frames"]
    fps_plan = 6
    total_frames = max(int(audio_duration * fps_plan), len(emotion_timeline) * 3)
    current = 0
    frames_per_emotion = max(1, total_frames // max(1, len(emotion_timeline)))

    for emotion in emotion_timeline:
        prompt = build_prompt(characters, emotion)
        print(f"emotion={emotion} prompt={prompt}")
        for _ in range(frames_per_emotion):
            if current >= total_frames:
                break
            frame_path = os.path.join(frames_dir, f"{record_id}_{current:04d}.png")
            try:
                with torch.inference_mode():
                    out = base_pipe(
                        prompt=prompt,
                        height=image_size,
                        width=image_size,
                        num_inference_steps=1,
                        guidance_scale=0.0,
                    )
                    out.images[0].save(frame_path)
                current += 1
                if current % 10 == 0:
                    clear_gpu_memory()
            except Exception as e:
                print(f"Frame {current} error: {e}")
                continue
    clear_gpu_memory()
    return current


def create_video_from_frames(record_id, audio_path, total_frames, audio_duration, fps=24):
    d = job_work_dirs(record_id)
    video_path = os.path.join(d["videos"], f"{record_id}.mp4")
    out_fps = max(24, min(total_frames / max(audio_duration, 0.1), 30))
    frame_pattern = os.path.join(d["frames"], f"{record_id}_%04d.png")
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(out_fps),
        "-i", frame_pattern,
        "-i", audio_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",
        video_path,
    ]
    subprocess.run(cmd, check=True)
    return video_path


def process_job_legacy(job_data):
    record_id = job_data["recordId"]
    characters = job_data.get("characters") or job_data.get("legacyCharacters") or []
    if characters and isinstance(characters[0], dict):
        characters = [c.get("referencePrompt", c.get("name", "")) for c in characters]
    emotions = job_data.get("emotions") or []
    narration = job_data.get("narration") or ""
    lang = job_data.get("language", "hi")

    print(f"LEGACY START {record_id}")
    audio_path, duration = generate_audio(record_id, narration, lang)
    total_frames = generate_frames(record_id, characters, emotions, duration)
    video = create_video_from_frames(record_id, audio_path, total_frames, duration)
    return {
        "recordId": record_id,
        "video": video,
        "audio": audio_path,
        "audioDuration": duration,
    }


def process_job(job_data):
    version = job_data.get("version", 1)
    beats = job_data.get("beats") or []

    if version >= 2 and len(beats) > 0:
        try:
            from pipeline.process_v2 import process_job_v2
            return process_job_v2(job_data)
        except ImportError:
            # Ensure package on path when running from repo layout
            import sys
            kaggle_dir = os.path.join(os.getcwd(), "cloud-jobs", "kaggle")
            if kaggle_dir not in sys.path:
                sys.path.insert(0, kaggle_dir)
            from pipeline.process_v2 import process_job_v2
            return process_job_v2(job_data)

    return process_job_legacy(job_data)


if __name__ == "__main__":
    print("Ready")
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
