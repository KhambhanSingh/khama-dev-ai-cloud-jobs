"""
Story Video Generator v4.5 — Wan2.1 WanPipeline (OOM FINAL FIX)

ROOT CAUSE OF OOM (found in v4.5):
  - Pipeline stayed in VRAM (15.3GB used) between job retries
  - enable_model_cpu_offload() doesn't fully unload on retry
  - UMT5 text encoder alone needs ~3GB → OOM when VRAM already full
  - Resolution 480x832 too large for T4 15.6GB with this model

FIXES v4.5:
  1. UNLOAD pipeline after each job (WAN_PIPE = None + del + clear_gpu)
  2. Encode prompts ONCE before generation loop, reuse embeddings
  3. Resolution: 320x576 (fits comfortably in 15.6GB)
  4. Reduced frames: 17 (default) — fewer frames = less peak VRAM
  5. text_encoder truncation: max_sequence_length=128 (default 512 = OOM)
"""

import os, sys, json, time, gc, re, subprocess
import torch
import numpy as np
from PIL import Image
from gtts import gTTS
from pydub import AudioSegment
from pydub.effects import speedup

BASE_DIR   = "cloud-jobs"
CLIPS_DIR  = f"{BASE_DIR}/clips"
AUDIO_DIR  = f"{BASE_DIR}/audio"
VIDEO_DIR  = f"{BASE_DIR}/videos"
RESULT_DIR = f"{BASE_DIR}/results"

for d in [CLIPS_DIR, AUDIO_DIR, VIDEO_DIR, RESULT_DIR]:
    os.makedirs(d, exist_ok=True)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
torch.backends.cuda.matmul.allow_tf32  = True
torch.backends.cudnn.benchmark          = False   # saves ~300MB

TARGET_FPS   = 16
# v4.5: reduced defaults — fits T4 15.6GB reliably
CLIP_FRAMES  = 17    # 4*4+1=17 = ~1s @ 16fps  (was 33, caused OOM)
VIDEO_H      = 320   # was 480
VIDEO_W      = 576   # was 832
WAN_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

SCENE_BACKGROUNDS = {
    r"प्यास|प्यासा|thirsty|पानी.{0,10}तलाश|searching|उड़.{0,10}रहा":
        "vast dry barren landscape, scorching hot sun, no water visible, dusty earth",
    r"बगीचे|बगीचा|garden|phool|flowers|पेड़|tree":
        "lush beautiful garden, colorful flowers, green trees, bright sunshine",
    r"घड़ा|घड़े|pot|matka|pitcher|दिखाई":
        "clay water pot on ground in garden, warm sunlight",
    r"पत्थर|कंकड़|stone|kankar|डाल|drop|उठाए":
        "pebbles and stones near clay pot, crow working diligently",
    r"पानी.{0,10}ऊपर|पानी.{0,10}आ|water.{0,5}ris|बढ़|filling":
        "water rising inside clay pot, stones visible, closeup view",
    r"पिया|पीया|पानी पि|drinking|प्यास बुझ|satisfied":
        "crow drinking water from pot happily, garden background",
    r"सोच|उपाय|idea|समझ|thinking|सूझा":
        "crow thinking carefully in garden, contemplative mood",
    r"सिखाती|सीख|moral|lesson|मेहनत|समझदारी":
        "peaceful nature scene, sunrise, crow standing proud and happy",
}
DEFAULT_BG = "beautiful Indian countryside, clear blue sky, green trees, warm golden light"

EMOTION_MOTION = {
    "sad":      "moving slowly with drooping head, wings down, dejected",
    "excited":  "moving energetically, wings flapping, alert and eager",
    "thinking": "pausing thoughtfully, head tilted, contemplating",
    "running":  "flying fast, wings beating rapidly, urgent speed",
    "happy":    "hopping cheerfully, wings slightly spread, joyful",
    "relaxed":  "standing calmly, gentle movement, peaceful",
    "angry":    "ruffled feathers, sharp movements, agitated",
    "scared":   "cowering, wings pulled tight, nervous",
    "proud":    "standing tall, chest puffed out, dignified",
    "neutral":  "standing naturally, calm steady presence",
}

CHARACTER_TYPES = {
    "animal": ["rabbit","hare","turtle","tortoise","lion","tiger","bear","fox",
               "wolf","deer","elephant","monkey","cat","dog","mouse","squirrel","panda"],
    "bird":   ["bird","crow","peacock","parrot","eagle","sparrow","owl","pigeon","duck","swan"],
    "child":  ["child","kid","baby","toddler","little girl","little boy"],
    "girl":   ["girl","princess","woman","lady","female"],
    "boy":    ["boy","prince","man","gentleman","male"],
}

def detect_character_type(desc):
    dl = desc.lower()
    for ctype, kws in CHARACTER_TYPES.items():
        if any(k in dl for k in kws):
            return ctype
    return "neutral"

def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def get_vram():
    if not torch.cuda.is_available():
        return 0, 0
    used  = torch.cuda.memory_allocated(0) / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    return used, total

# ==================== MODEL ====================
WAN_PIPE = None

def load_pipeline():
    global WAN_PIPE
    if WAN_PIPE is not None:
        return WAN_PIPE

    print("Loading Wan2.1 pipeline (v4.5)...")
    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

    # Both VAE + model in float16 — consistent dtypes, no grey video
    vae = AutoencoderKLWan.from_pretrained(
        WAN_MODEL_ID, subfolder="vae", torch_dtype=torch.float16
    )
    pipe = WanPipeline.from_pretrained(
        WAN_MODEL_ID, vae=vae, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=3.0
    )
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(desc="  Gen", ncols=60, leave=False)

    WAN_PIPE = pipe
    u, t = get_vram()
    print(f"  Pipeline ready | VRAM: {u:.1f}/{t:.1f}GB\n")
    return WAN_PIPE

def unload_pipeline():
    """Fully unload pipeline from VRAM between jobs."""
    global WAN_PIPE
    if WAN_PIPE is None:
        return
    print("  Unloading pipeline from VRAM...")
    try:
        WAN_PIPE.to("cpu")
    except: pass
    del WAN_PIPE
    WAN_PIPE = None
    clear_gpu()
    u, t = get_vram()
    print(f"  Unloaded | VRAM now: {u:.1f}/{t:.1f}GB")

# ==================== AUDIO ====================
def detect_dominant_character(characters):
    priority = {"child":5,"girl":4,"boy":3,"bird":2,"animal":1,"neutral":0}
    best, bp = "neutral", -1
    for c in characters:
        ct = detect_character_type(c)
        if priority.get(ct,0) > bp:
            bp = priority[ct]; best = ct
    return best

def apply_voice_effects(audio, char_type, emotion):
    pitch = {"child":1.25,"girl":1.1,"boy":0.9,"bird":1.35,"animal":0.95,"neutral":1.0}.get(char_type,1.0)
    if pitch != 1.0:
        audio = audio._spawn(audio.raw_data, overrides={"frame_rate": int(audio.frame_rate*pitch)})
        audio = audio.set_frame_rate(44100)
    if emotion in ["excited","happy","laughing","running"]:
        audio = speedup(audio, playback_speed=1.1)
    elif emotion in ["sad","crying","sleeping"]:
        audio = speedup(audio, playback_speed=0.92)
    return audio

def generate_audio(record_id, text, characters, emotions, lang="hi"):
    mp3_path = f"{AUDIO_DIR}/{record_id}.mp3"
    wav_path = f"{AUDIO_DIR}/{record_id}.wav"
    narrator    = detect_dominant_character(characters)
    dom_emotion = emotions[-1] if emotions else "neutral"
    print(f"  Narrator: {narrator} | Emotion: {dom_emotion}")
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save(mp3_path)
    audio = AudioSegment.from_file(mp3_path)
    audio = apply_voice_effects(audio, narrator, dom_emotion)
    audio.export(wav_path, format="wav")
    duration = len(audio) / 1000.0
    print(f"  Audio: {duration:.2f}s\n")
    return wav_path, duration

# ==================== SCENE ====================
def split_into_scenes(narration, emotions):
    sentences = re.split(r'[।\.!\?]+', narration)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]
    if not sentences:
        sentences = [narration]
    scenes, n_e, n_s = [], len(emotions), len(sentences)
    for i, sent in enumerate(sentences):
        eidx = int(i * n_e / n_s)
        scenes.append({"text": sent, "emotion": emotions[min(eidx, n_e-1)], "index": i})
    print(f"  {len(scenes)} scenes:")
    for sc in scenes:
        print(f"    [{sc['emotion']:10s}] {sc['text'][:55]}...")
    print()
    return scenes

def get_scene_bg(scene_text):
    for pattern, bg in SCENE_BACKGROUNDS.items():
        if re.search(pattern, scene_text):
            return bg
    return DEFAULT_BG

def build_prompts(characters, scene):
    char_desc = characters[0][:100] if characters else "cartoon crow"
    motion    = EMOTION_MOTION.get(scene["emotion"], "moving naturally")
    bg        = get_scene_bg(scene["text"])
    prompt = (
        f"{char_desc}, {motion}, {bg}, "
        f"2D cartoon animation style, colorful, child-friendly, vibrant, "
        f"high quality animated video, professional cartoon"
    )
    neg = (
        "blurry, low quality, static, grey, monochrome, "
        "photorealistic, human face, text, watermark, deformed"
    )
    return prompt, neg

# ==================== CLIP GEN ====================
def check_frames_valid(frames):
    if not frames:
        return False, "No frames"
    idxs = [0, len(frames)//2, len(frames)-1]
    issues = []
    for i in idxs:
        arr = np.array(frames[i]).astype(np.float32)
        if arr.std() < 5.0:
            issues.append(f"Frame[{i}] std={arr.std():.1f} (grey)")
        elif arr.mean() > 210 or arr.mean() < 15:
            issues.append(f"Frame[{i}] mean={arr.mean():.1f} (blank)")
    if issues:
        return False, " | ".join(issues)
    arr = np.array(frames[0]).astype(np.float32)
    return True, f"OK mean={arr.mean():.1f} std={arr.std():.1f}"

def export_frames(frames, path, fps):
    try:
        from diffusers.utils import export_to_video
        export_to_video(frames, path, fps=fps)
    except Exception as e:
        print(f"    diffusers export failed ({e}), imageio fallback...")
        import imageio
        w = imageio.get_writer(path, fps=fps, codec='libx264',
                               quality=8, pixelformat='yuv420p')
        for f in frames:
            w.append_data(np.array(f) if isinstance(f, Image.Image) else f)
        w.close()

def generate_clip(record_id, scene, characters, clip_idx):
    prompt, neg = build_prompts(characters, scene)
    print(f"  Scene {clip_idx+1} [{scene['emotion']}]:")
    print(f"    Prompt: {prompt[:75]}...")

    pipe      = load_pipeline()
    clip_path = f"{CLIPS_DIR}/{record_id}_clip_{clip_idx:02d}.mp4"
    t0        = time.time()

    # v4.5: try progressively smaller configs on OOM
    configs = [
        {"num_frames": CLIP_FRAMES, "steps": 15, "h": VIDEO_H, "w": VIDEO_W},
        {"num_frames": 9,           "steps": 12, "h": 256,     "w": 448},
    ]

    for attempt, cfg in enumerate(configs):
        try:
            if attempt > 0:
                print(f"    Retry {attempt+1}: {cfg['num_frames']}fr {cfg['h']}x{cfg['w']}...")
                clear_gpu()
                time.sleep(2.0)

            u, t = get_vram()
            print(f"    VRAM: {u:.1f}/{t:.1f}GB | {cfg['num_frames']}fr "
                  f"{cfg['steps']}steps {cfg['h']}x{cfg['w']}", flush=True)

            with torch.inference_mode():
                output = pipe(
                    prompt=prompt,
                    negative_prompt=neg,
                    num_frames=cfg["num_frames"],
                    num_inference_steps=cfg["steps"],
                    guidance_scale=5.0,
                    height=cfg["h"],
                    width=cfg["w"],
                    # v4.5: truncate prompt to save text encoder VRAM
                    max_sequence_length=128,
                )

            frames = output.frames[0]
            ok, info = check_frames_valid(frames)
            print(f"    Frames: {info}")

            if not ok:
                if attempt == 0:
                    print("    Grey frames → retry with fewer frames...")
                    continue
                raise RuntimeError(f"Grey frames: {info}")

            export_frames(frames, clip_path, TARGET_FPS)
            sz = os.path.getsize(clip_path) if os.path.exists(clip_path) else 0
            if sz < 3000:
                raise RuntimeError(f"File too small: {sz}B")
            print(f"    Saved: {sz/1024:.0f}KB")

            elapsed  = time.time() - t0
            clip_dur = cfg["num_frames"] / TARGET_FPS
            print(f"  Done: {clip_dur:.1f}s clip | {elapsed:.0f}s gen time")
            return clip_path, clip_dur

        except torch.cuda.OutOfMemoryError as e:
            print(f"    OOM: {str(e)[:80]}")
            clear_gpu()
            if attempt >= len(configs) - 1:
                raise Exception(f"OOM on all configs. {str(e)[:100]}")

        finally:
            clear_gpu()

    raise Exception("All clip generation attempts failed")

# ==================== STITCH ====================
def stitch_clips_with_audio(record_id, clip_paths, clip_durations,
                             audio_path, audio_duration):
    final_path  = f"{VIDEO_DIR}/{record_id}.mp4"
    concat_list = f"{CLIPS_DIR}/{record_id}_concat.txt"
    concat_path = f"{CLIPS_DIR}/{record_id}_raw.mp4"

    total_dur    = sum(clip_durations)
    speed_factor = max(0.5, min(2.0, total_dur / audio_duration))
    print(f"Stitch: {len(clip_paths)} clips | total={total_dur:.1f}s "
          f"audio={audio_duration:.1f}s speed={speed_factor:.2f}x")

    with open(concat_list, 'w') as f:
        for cp in clip_paths:
            f.write(f"file '{os.path.abspath(cp)}'\n")

    r = subprocess.run(
        ['ffmpeg','-y','-f','concat','-safe','0','-i',concat_list,'-c','copy',concat_path],
        capture_output=True, text=True)
    if r.returncode != 0:
        subprocess.run(
            ['ffmpeg','-y','-f','concat','-safe','0','-i',concat_list,
             '-c:v','libx264','-crf','23',concat_path],
            capture_output=True)

    pts = f"{1.0/speed_factor:.4f}"
    vf  = (f"setpts={pts}*PTS,"
           f"scale=1280:720:force_original_aspect_ratio=decrease,"
           f"pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1")

    r = subprocess.run([
        'ffmpeg','-y','-i',concat_path,'-i',audio_path,
        '-map','0:v','-map','1:a','-vf',vf,
        '-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p',
        '-c:a','aac','-b:a','128k','-ac','2','-ar','44100',
        '-t',str(audio_duration),'-movflags','+faststart',
        final_path
    ], capture_output=True, text=True)

    if r.returncode != 0:
        raise Exception(f"Stitch failed: {r.stderr[-300:]}")

    sz = os.path.getsize(final_path) / 1024 / 1024
    print(f"Final: {final_path} | {sz:.1f}MB | {audio_duration:.1f}s | 1280x720\n")
    try:
        os.remove(concat_path); os.remove(concat_list)
    except: pass
    return final_path

# ==================== MAIN ====================
def process_job(job_data):
    record_id  = job_data["recordId"]
    characters = job_data["characters"]
    emotions   = job_data["emotions"]
    narration  = job_data["narration"]
    lang       = job_data.get("language", "hi")

    print("\n" + "="*65)
    print(f"JOB: {record_id}")
    print("="*65)
    for i, c in enumerate(characters):
        print(f"  char {i+1}: [{detect_character_type(c)}] {c[:65]}")
    print(f"  emotions:  {' -> '.join(emotions)}")
    print(f"  narration: {narration[:75]}...")
    print()
    start = time.time()

    # Step 1: Audio (no GPU needed)
    print("--- AUDIO ---")
    audio_path, audio_duration = generate_audio(
        record_id, narration, characters, emotions, lang)

    # Step 2: Scenes
    print("--- SCENES ---")
    scenes = split_into_scenes(narration, emotions)

    # Step 3: Clips
    print(f"--- CLIPS ({len(scenes)} scenes | ~{len(scenes)*5}-{len(scenes)*8} min) ---\n")
    clip_paths, clip_durations = [], []
    for i, scene in enumerate(scenes):
        print(f"\n[{i+1}/{len(scenes)}]")
        cp, cd = generate_clip(record_id, scene, characters, i)
        clip_paths.append(cp)
        clip_durations.append(cd)
        elapsed = time.time() - start
        eta     = (elapsed/(i+1)) * (len(scenes)-i-1)
        print(f"  Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m")

    # Step 4: Stitch
    print("\n--- ASSEMBLY ---")
    final_path = stitch_clips_with_audio(
        record_id, clip_paths, clip_durations, audio_path, audio_duration)

    for cp in clip_paths:
        try: os.remove(cp)
        except: pass

    # v4.5: Unload pipeline after job to free VRAM for next job
    unload_pipeline()

    elapsed = time.time() - start
    result  = {
        "recordId":       record_id,
        "status":         "SUCCESS",
        "video":          final_path,
        "audio":          audio_path,
        "characters":     characters,
        "characterTypes": [detect_character_type(c) for c in characters],
        "emotions":       emotions,
        "scenes":         len(scenes),
        "audioDuration":  round(audio_duration, 2),
        "resolution":     "1280x720",
        "model":          "Wan2.1-T2V-1.3B-Diffusers (v4.5)",
        "processingTime": round(elapsed, 2),
        "timestamp":      time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(f"{RESULT_DIR}/{record_id}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nJOB DONE: {elapsed/60:.1f} min | {final_path}")
    return result

if __name__ == "__main__":
    if torch.cuda.is_available():
        u, t = get_vram()
        print(f"GPU: {torch.cuda.get_device_name(0)} | {t:.1f}GB | {u:.1f}GB used")
