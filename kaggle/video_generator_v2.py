"""
Story Video Generator v4.4 — Wan2.1 WanPipeline (GREY VIDEO FIX)
- FIXED v4.4: Grey/noisy video — VAE dtype mismatch fix
  - VAE now float16 (same as model) to prevent decode failure
  - Back to enable_model_cpu_offload() but WITHOUT autocast (safe combo)
  - Added output.frames debug check before export
  - imageio fallback export
- FIXED v4.3: CUBLAS_STATUS_ALLOC_FAILED — removed autocast wrapper
- WanPipeline from diffusers (not AutoPipelineForText2Video)
- Model: Wan-AI/Wan2.1-T2V-1.3B-Diffusers
"""

import os, sys, json, time, gc, re, glob, subprocess, shutil
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

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
torch.backends.cuda.matmul.allow_tf32 = True

TARGET_FPS   = 16
CLIP_FRAMES  = 33
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
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

def get_gpu_memory_usage():
    if not torch.cuda.is_available():
        return 0, 0
    props = torch.cuda.get_device_properties(0)
    allocated = torch.cuda.memory_allocated(0) / 1e9
    total = props.total_memory / 1e9
    return allocated, total

def cublas_prewarm():
    if not torch.cuda.is_available():
        return
    try:
        _a = torch.zeros(32, 32, device='cuda', dtype=torch.float16)
        _b = torch.zeros(32, 32, device='cuda', dtype=torch.float16)
        _c = torch.matmul(_a, _b)
        del _a, _b, _c
        torch.cuda.synchronize()
        clear_gpu()
        print("  cuBLAS pre-warm: OK")
    except Exception as e:
        print(f"  cuBLAS pre-warm warning: {e}")

def check_frames_valid(frames):
    """
    Detect grey/noisy output frames before export.
    Grey video: mean ~128, std < 5 (uniform grey)
    Noise video: std very high but no structure — catch with low color variance
    Returns (is_valid, debug_string)
    """
    if not frames or len(frames) == 0:
        return False, "No frames returned"

    check_idxs = [0, len(frames)//2, len(frames)-1]
    issues = []

    for idx in check_idxs:
        frame = frames[idx]
        arr = np.array(frame).astype(np.float32)
        mean_val = arr.mean()
        std_val  = arr.std()

        if std_val < 5.0:
            issues.append(f"Frame[{idx}] std={std_val:.1f} (grey/static)")
        elif mean_val > 210 or mean_val < 15:
            issues.append(f"Frame[{idx}] mean={mean_val:.1f} (washed/black)")

    if issues:
        return False, " | ".join(issues)

    arr = np.array(frames[0]).astype(np.float32)
    return True, f"OK mean={arr.mean():.1f} std={arr.std():.1f} n={len(frames)}"

def _export_via_imageio(frames, output_path, fps):
    """Fallback video export using imageio."""
    import imageio
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264',
                                quality=8, pixelformat='yuv420p')
    for frame in frames:
        if isinstance(frame, Image.Image):
            frame = np.array(frame)
        writer.append_data(frame)
    writer.close()

# ==================== MODEL LOADING ====================
WAN_PIPE = None

def get_wan_pipeline():
    """
    Load Wan2.1 pipeline with grey-video fix.

    ROOT CAUSE OF GREY VIDEO:
      VAE was float32, diffusion model output latents are float16.
      On decode, the float32 VAE receives float16 tensors → NaN/Inf
      → grey noise rendered by ffmpeg.

    FIX: Both VAE and model in float16 — consistent dtypes, no NaN on decode.
    At 480P resolution float16 VAE quality is fine (float32 only needed for 720P+).

    CUBLAS FIX (v4.3): No autocast wrapper in inference call.
    """
    global WAN_PIPE
    if WAN_PIPE is not None:
        return WAN_PIPE

    print("Loading Wan2.1-T2V-1.3B-Diffusers (v4.4: grey-video-fixed)...")
    print("  (First run: ~5 min to download weights)")

    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

    # FIX v4.4: float16 VAE — matches model dtype, prevents NaN decode → grey video
    vae = AutoencoderKLWan.from_pretrained(
        WAN_MODEL_ID,
        subfolder="vae",
        torch_dtype=torch.float16
    )

    pipe = WanPipeline.from_pretrained(
        WAN_MODEL_ID,
        vae=vae,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config,
        flow_shift=3.0
    )

    pipe.enable_model_cpu_offload()
    print("  CPU offload: model-level")

    try:
        import xformers
        if hasattr(pipe, 'unet'):
            pipe.unet.set_use_memory_efficient_attention_xformers(True)
            print("  xFormers: enabled")
    except ImportError:
        print("  xFormers: not installed (optional)")
    except Exception as e:
        print(f"  xFormers: n/a ({str(e)[:40]})")

    pipe.set_progress_bar_config(desc="  Generating", ncols=60, leave=False)

    print("  cuBLAS pre-warm...")
    cublas_prewarm()

    WAN_PIPE = pipe
    print("Pipeline ready!\n")
    return WAN_PIPE

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
    narrator = detect_dominant_character(characters)
    dom_emotion = emotions[-1] if emotions else "neutral"
    print(f"Narrator: {narrator} | Emotion: {dom_emotion}")
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save(mp3_path)
    audio = AudioSegment.from_file(mp3_path)
    audio = apply_voice_effects(audio, narrator, dom_emotion)
    audio.export(wav_path, format="wav")
    duration = len(audio) / 1000.0
    print(f"  Audio duration: {duration:.2f}s\n")
    return wav_path, duration

# ==================== SCENE PARSING ====================
def split_into_scenes(narration, emotions):
    sentences = re.split(r'[।\.!\?]+', narration)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]
    if not sentences:
        sentences = [narration]
    scenes = []
    n_e, n_s = len(emotions), len(sentences)
    for i, sent in enumerate(sentences):
        eidx = int(i * n_e / n_s)
        scenes.append({"text": sent, "emotion": emotions[min(eidx, n_e-1)], "index": i})
    print(f"{len(scenes)} scenes planned:")
    for sc in scenes:
        print(f"  [{sc['emotion']:10s}] {sc['text'][:60]}...")
    print()
    return scenes

def get_scene_bg(scene_text):
    for pattern, bg in SCENE_BACKGROUNDS.items():
        if re.search(pattern, scene_text):
            return bg
    return DEFAULT_BG

def build_wan_prompt(characters, scene):
    char_desc = characters[0][:100] if characters else "cartoon crow"
    motion    = EMOTION_MOTION.get(scene["emotion"], "moving naturally")
    bg        = get_scene_bg(scene["text"])
    prompt = (
        f"{char_desc}, {motion}, "
        f"{bg}, "
        f"2D cartoon animation style, colorful, child-friendly, vibrant, smooth animation, "
        f"high quality animated video, smooth movement, professional cartoon"
    )
    neg = "blurry, low quality, static, grey, monochrome, photorealistic, human face, text, watermark, deformed, ugly"
    return prompt, neg

# ==================== CLIP GENERATION ====================
def generate_clip(record_id, scene, characters, clip_idx):
    prompt, neg_prompt = build_wan_prompt(characters, scene)
    print(f"  Clip {clip_idx+1} [{scene['emotion']}]: {prompt[:80]}...")

    pipe = get_wan_pipeline()
    clip_path = f"{CLIPS_DIR}/{record_id}_clip_{clip_idx:02d}.mp4"
    t0 = time.time()

    n_frames = CLIP_FRAMES
    max_attempts = 2

    for attempt in range(max_attempts):
        try:
            if attempt > 0:
                print(f"    Attempt {attempt+1}: cleanup + retry...")
                clear_gpu()
                time.sleep(1.0)

            used, total = get_gpu_memory_usage()
            print(f"    VRAM: {used:.1f}/{total:.1f}GB", flush=True)

            steps    = 20 if attempt > 0 else 15   # more steps on retry helps quality
            guidance = 5.0

            print(f"    Generating {n_frames} frames ({steps} steps, cfg={guidance})...", flush=True)

            # NO autocast wrapper — causes cuBLAS alloc failure with CPU offload
            with torch.inference_mode():
                output = pipe(
                    prompt=prompt,
                    negative_prompt=neg_prompt,
                    num_frames=n_frames,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    height=480,
                    width=832,
                )

            frames = output.frames[0]

            # Validate — catch grey output before wasting time on export
            is_valid, debug_info = check_frames_valid(frames)
            print(f"    Frame check: {debug_info}")

            if not is_valid:
                if attempt == 0:
                    print(f"    Grey frames → retrying with more steps...")
                    n_frames = CLIP_FRAMES
                    continue
                else:
                    raise RuntimeError(f"Grey frames persisted after retry: {debug_info}")

            # Export
            print(f"    Exporting...", flush=True)
            try:
                from diffusers.utils import export_to_video
                export_to_video(frames, clip_path, fps=TARGET_FPS)
            except Exception as ex:
                print(f"    diffusers export failed ({ex}), imageio fallback...")
                _export_via_imageio(frames, clip_path, TARGET_FPS)

            # Sanity check file
            sz = os.path.getsize(clip_path) if os.path.exists(clip_path) else 0
            if sz < 5000:
                raise RuntimeError(f"Exported file too small ({sz}B)")
            print(f"    Export OK: {sz/1024:.0f}KB", flush=True)
            break

        except torch.cuda.OutOfMemoryError as e:
            if attempt == 0:
                print(f"    OOM → retrying with 17 frames...")
                n_frames = 17
            else:
                raise Exception(f"OOM persisted: {str(e)[:100]}")

        finally:
            clear_gpu()

    clear_gpu()
    clip_dur = n_frames / TARGET_FPS
    size_mb  = os.path.getsize(clip_path) / 1024 / 1024 if os.path.exists(clip_path) else 0
    elapsed  = time.time() - t0
    print(f"  Done: {clip_dur:.1f}s | {size_mb:.1f}MB | {elapsed:.0f}s")
    return clip_path, clip_dur

# ==================== STITCH + AUDIO ====================
def stitch_clips_with_audio(record_id, clip_paths, clip_durations, audio_path, audio_duration):
    final_path  = f"{VIDEO_DIR}/{record_id}.mp4"
    concat_list = f"{CLIPS_DIR}/{record_id}_concat.txt"
    concat_path = f"{CLIPS_DIR}/{record_id}_raw.mp4"

    total_dur    = sum(clip_durations)
    speed_factor = max(0.5, min(2.0, total_dur / audio_duration))

    print(f"Stitching {len(clip_paths)} clips:")
    print(f"  Clip total: {total_dur:.1f}s | Audio: {audio_duration:.1f}s | Speed: {speed_factor:.2f}x")

    with open(concat_list, 'w') as f:
        for cp in clip_paths:
            f.write(f"file '{os.path.abspath(cp)}'\n")

    r = subprocess.run(
        ['ffmpeg','-y','-f','concat','-safe','0','-i',concat_list,'-c','copy',concat_path],
        capture_output=True, text=True
    )
    if r.returncode != 0:
        subprocess.run(
            ['ffmpeg','-y','-f','concat','-safe','0','-i',concat_list,
             '-c:v','libx264','-crf','20',concat_path],
            capture_output=True
        )

    pts = f"{1.0/speed_factor:.4f}"
    vf  = (f"setpts={pts}*PTS,"
           f"scale=1280:720:force_original_aspect_ratio=decrease,"
           f"pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1")

    r = subprocess.run([
        'ffmpeg','-y','-i',concat_path,'-i',audio_path,
        '-map','0:v','-map','1:a',
        '-vf', vf,
        '-c:v','libx264','-preset','medium','-crf','18','-pix_fmt','yuv420p',
        '-c:a','aac','-b:a','192k','-ac','2','-ar','44100',
        '-t', str(audio_duration),
        '-movflags','+faststart',
        final_path
    ], capture_output=True, text=True)

    if r.returncode != 0:
        raise Exception(f"Stitch failed: {r.stderr[-300:]}")

    size_mb = os.path.getsize(final_path) / 1024 / 1024
    print(f"Final video: {final_path} | {size_mb:.1f}MB | {audio_duration:.1f}s | 1280x720\n")

    try:
        os.remove(concat_path)
        os.remove(concat_list)
    except: pass
    return final_path

# ==================== MAIN ====================
def process_job(job_data):
    record_id  = job_data["recordId"]
    characters = job_data["characters"]
    emotions   = job_data["emotions"]
    narration  = job_data["narration"]
    lang       = job_data.get("language", "hi")

    print("\n" + "="*70)
    print(f"WAN2.1 JOB: {record_id}")
    print("="*70)
    for i,c in enumerate(characters):
        print(f"  char {i+1}: [{detect_character_type(c)}] {c[:70]}")
    print(f"  emotions: {' -> '.join(emotions)}")
    print(f"  narration: {narration[:80]}...")
    print()
    start = time.time()

    print("--- STEP 1: AUDIO ---")
    audio_path, audio_duration = generate_audio(record_id, narration, characters, emotions, lang)

    print("--- STEP 2: SCENE PLANNING ---")
    scenes = split_into_scenes(narration, emotions)

    print(f"--- STEP 3: WAN2.1 CLIPS ({len(scenes)} scenes) ---")
    print(f"  Est: {len(scenes)*10}-{len(scenes)*15} min\n")

    clip_paths, clip_durations = [], []
    for i, scene in enumerate(scenes):
        print(f"\nScene {i+1}/{len(scenes)}:")
        cp, cd = generate_clip(record_id, scene, characters, i)
        clip_paths.append(cp)
        clip_durations.append(cd)
        so_far = time.time() - start
        eta    = (so_far/(i+1)) * (len(scenes)-i-1)
        print(f"  Elapsed: {so_far/60:.1f}min | ETA: {eta/60:.1f}min")

    print("\n--- STEP 4: FINAL ASSEMBLY ---")
    final_path = stitch_clips_with_audio(
        record_id, clip_paths, clip_durations, audio_path, audio_duration
    )
    for cp in clip_paths:
        try: os.remove(cp)
        except: pass

    elapsed = time.time() - start
    result = {
        "recordId": record_id, "status": "SUCCESS",
        "video": final_path, "audio": audio_path,
        "characters": characters,
        "characterTypes": [detect_character_type(c) for c in characters],
        "emotions": emotions, "scenes": len(scenes),
        "audioDuration": round(audio_duration, 2),
        "resolution": "1280x720",
        "model": "Wan2.1-T2V-1.3B-Diffusers (v4.4 grey-fixed)",
        "processingTime": round(elapsed, 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(f"{RESULT_DIR}/{record_id}.json","w",encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"JOB DONE: {elapsed/60:.1f} min | {final_path}")
    return result

if __name__ == "__main__":
    if torch.cuda.is_available():
        used, total = get_gpu_memory_usage()
        print(f"GPU: {torch.cuda.get_device_name(0)} | {total:.1f}GB total | {used:.1f}GB in use")
