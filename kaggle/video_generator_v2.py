"""
Story Video Generator v5.1 — Wan2.1 WanPipeline

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPLETE BUG HISTORY & ROOT CAUSES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
v4.2/4.3/4.5  CUBLAS_STATUS_ALLOC_FAILED
              accelerate CPU-offload hooks intercept forward()
              and corrupt cuBLAS handle init on torch 2.10+cu128

v4.4          OOM at 15.3GB on retry
              Pipeline never unloaded between retries

v4.6          OOM at pipe.to("cuda")
              UMT5 text encoder is ~9.4GB (T5-XXL based).
              from_pretrained→CPU then .to("cuda") = 18.8GB peak → OOM

v5.0          NotImplementedError: aten::slow_conv3d_forward / CUDA
              torch.backends.cudnn.benchmark=False (added in v4.5) disables
              cuDNN conv3d. Kaggle torch 2.10+cu128 is a selective build where
              slow_conv3d (CPU fallback) is NOT registered for CUDA.
              WanTransformer3D.patch_embedding is Conv3d → crash at step 0.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
v5.1 FINAL ARCHITECTURE (all issues resolved)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. cudnn.enabled=True + cudnn.benchmark=True   → Conv3d works on CUDA
2. Text encoder on CPU → encoded → deleted     → no CUBLAS, no OOM
3. Only transformer+VAE moved to GPU (~3.8GB)  → fits in 15.6GB
4. prompt_embeds= passed directly              → text encoder never called
5. unload_video_pipeline() after each job      → no VRAM leak between jobs
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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
torch.backends.cuda.matmul.allow_tf32 = True

# v5.1 FIX: cuDNN MUST be enabled
# Kaggle torch 2.10+cu128 selective build: slow_conv3d NOT compiled for CUDA.
# cudnn.benchmark=False (set in v4.5) disabled cuDNN → crash at Conv3d step.
torch.backends.cudnn.enabled   = True   # required for Conv3d on CUDA
torch.backends.cudnn.benchmark = True   # use cuDNN kernels (NOT False)

TARGET_FPS   = 16
CLIP_FRAMES  = 17       # 4*4+1 = ~1.06s @ 16fps
VIDEO_H      = 320
VIDEO_W      = 576
WAN_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

# ── scene knowledge ──────────────────────────────────────────────
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

# ── utilities ────────────────────────────────────────────────────
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
        return 0.0, 0.0
    used  = torch.cuda.memory_allocated(0) / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    return used, total

def vram_str():
    u, t = get_vram()
    return f"{u:.1f}/{t:.1f}GB"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 1 — Text encoding on CPU (encoder deleted after)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def encode_prompts_on_cpu(prompts_pos, prompts_neg):
    """
    Run UMT5 entirely on CPU. Encode all prompts. Delete the encoder.

    Why CPU + float32:
      UMT5 in Wan2.1 is ~9.4GB in float16 — too large to fit on GPU
      alongside transformer+VAE during inference (~3.8GB). float32 on
      CPU prevents NaN in long-sequence attention. Output is cast to
      float16 for GPU inference compatibility.

    Why delete:
      All embeddings stored as small CPU tensors. The 9.4GB model
      is not needed again.
    """
    print("\n[PHASE 1: TEXT ENCODING — CPU only, encoder deleted after]")
    from transformers import AutoTokenizer, UMT5EncoderModel

    print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        WAN_MODEL_ID, subfolder="tokenizer"
    )

    print("  Loading UMT5 text encoder on CPU (float32)...")
    text_encoder = UMT5EncoderModel.from_pretrained(
        WAN_MODEL_ID,
        subfolder="text_encoder",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    text_encoder.eval()

    MAX_LEN = 128

    def _encode(texts):
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
        )
        with torch.inference_mode():
            out = text_encoder(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
            )
        return out.last_hidden_state.to(dtype=torch.float16).detach().cpu()

    print(f"  Encoding {len(prompts_pos)} positive + {len(prompts_neg)} negative prompts...")
    pos_embeds = [_encode([p]) for p in prompts_pos]
    neg_embeds = [_encode([n]) for n in prompts_neg]

    print("  Deleting text encoder from RAM...")
    del text_encoder, tokenizer
    gc.collect()
    print(f"  Done | VRAM: {vram_str()}\n")
    return pos_embeds, neg_embeds


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 2 — Video pipeline on GPU (transformer + VAE only)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VIDEO_PIPE = None

def load_video_pipeline():
    """
    Load transformer (~2.6GB) + VAE (~1.2GB) directly to GPU.
    text_encoder stays on CPU (unused — we pass prompt_embeds=).

    No accelerate hooks → no CUBLAS issue.
    cuDNN enabled (v5.1 fix) → Conv3d works on CUDA.
    Components moved individually → no double-memory OOM from .to("cuda").
    """
    global VIDEO_PIPE
    if VIDEO_PIPE is not None:
        return VIDEO_PIPE

    print("\n[PHASE 2: VIDEO PIPELINE — transformer+VAE on GPU]")
    clear_gpu()
    print(f"  VRAM before: {vram_str()}")

    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

    vae = AutoencoderKLWan.from_pretrained(
        WAN_MODEL_ID,
        subfolder="vae",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    pipe = WanPipeline.from_pretrained(
        WAN_MODEL_ID,
        vae=vae,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=3.0
    )

    # Move only video-gen components to GPU individually
    # (avoids pipe.to("cuda") which doubles peak memory)
    print("  Moving transformer → CUDA...")
    pipe.transformer = pipe.transformer.to("cuda")

    print("  Moving VAE → CUDA...")
    pipe.vae = pipe.vae.to("cuda")

    # text_encoder: keep on CPU — we pass prompt_embeds= so it's never called
    pipe.text_encoder = pipe.text_encoder.cpu()

    pipe.enable_attention_slicing(1)
    try:
        pipe.enable_vae_slicing()
        print("  VAE slicing: enabled")
    except Exception:
        pass

    pipe.set_progress_bar_config(desc="  Generating", ncols=70, leave=False)

    VIDEO_PIPE = pipe
    print(f"  Pipeline ready | VRAM: {vram_str()}\n")
    return VIDEO_PIPE


def unload_video_pipeline():
    global VIDEO_PIPE
    if VIDEO_PIPE is None:
        return
    print("  Unloading pipeline...")
    try:
        VIDEO_PIPE.transformer.to("cpu")
        VIDEO_PIPE.vae.to("cpu")
    except Exception:
        pass
    del VIDEO_PIPE
    VIDEO_PIPE = None
    clear_gpu()
    print(f"  Unloaded | VRAM: {vram_str()}")


# ── audio ─────────────────────────────────────────────────────────
def detect_dominant_character(characters):
    priority = {"child":5,"girl":4,"boy":3,"bird":2,"animal":1,"neutral":0}
    best, bp = "neutral", -1
    for c in characters:
        ct = detect_character_type(c)
        if priority.get(ct, 0) > bp:
            bp = priority[ct]; best = ct
    return best

def apply_voice_effects(audio, char_type, emotion):
    pitch = {
        "child":1.25,"girl":1.1,"boy":0.9,
        "bird":1.35,"animal":0.95,"neutral":1.0
    }.get(char_type, 1.0)
    if pitch != 1.0:
        audio = audio._spawn(
            audio.raw_data,
            overrides={"frame_rate": int(audio.frame_rate * pitch)}
        )
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
    print(f"  Duration: {duration:.2f}s\n")
    return wav_path, duration


# ── scene planning ────────────────────────────────────────────────
def split_into_scenes(narration, emotions):
    sentences = re.split(r'[।\.!\?]+', narration)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]
    if not sentences:
        sentences = [narration]
    scenes, n_e, n_s = [], len(emotions), len(sentences)
    for i, sent in enumerate(sentences):
        eidx = int(i * n_e / n_s)
        scenes.append({
            "text":    sent,
            "emotion": emotions[min(eidx, n_e - 1)],
            "index":   i,
        })
    print(f"  {len(scenes)} scenes:")
    for sc in scenes:
        print(f"    [{sc['emotion']:10s}] {sc['text'][:55]}...")
    print()
    return scenes

def get_scene_bg(text):
    for pattern, bg in SCENE_BACKGROUNDS.items():
        if re.search(pattern, text):
            return bg
    return DEFAULT_BG

def build_scene_prompt(characters, scene):
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
        "photorealistic, human face, text, watermark, deformed, ugly"
    )
    return prompt, neg


# ── frame quality check ───────────────────────────────────────────
def check_frames_valid(frames):
    if not frames:
        return False, "No frames returned"
    idxs   = [0, len(frames) // 2, len(frames) - 1]
    issues = []
    for i in idxs:
        arr = np.array(frames[i]).astype(np.float32)
        if arr.std() < 5.0:
            issues.append(f"F[{i}] std={arr.std():.1f}(grey)")
        elif arr.mean() > 210 or arr.mean() < 15:
            issues.append(f"F[{i}] mean={arr.mean():.1f}(blank)")
    if issues:
        return False, " | ".join(issues)
    arr = np.array(frames[0]).astype(np.float32)
    return True, f"OK mean={arr.mean():.1f} std={arr.std():.1f} n={len(frames)}"

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


# ── clip generation ───────────────────────────────────────────────
def generate_clip(record_id, clip_idx, pos_embed, neg_embed):
    """
    Generate one video clip using pre-computed CPU embeddings.
    Moves embeddings to GPU only during the call.
    No text encoder → no CUBLAS. cuDNN on → Conv3d works.
    """
    pipe      = load_video_pipeline()
    clip_path = f"{CLIPS_DIR}/{record_id}_clip_{clip_idx:02d}.mp4"
    t0        = time.time()

    configs = [
        {"num_frames": CLIP_FRAMES, "steps": 15, "h": VIDEO_H, "w": VIDEO_W},
        {"num_frames": 9,           "steps": 12, "h": 256,     "w": 448},
    ]

    for attempt, cfg in enumerate(configs):
        pos_gpu = neg_gpu = None
        try:
            if attempt > 0:
                print(f"    Fallback → {cfg['num_frames']}fr {cfg['h']}×{cfg['w']}...")
                clear_gpu()
                time.sleep(1.0)

            pos_gpu = pos_embed.to("cuda")
            neg_gpu = neg_embed.to("cuda")

            print(f"    VRAM: {vram_str()} | "
                  f"{cfg['num_frames']}fr {cfg['steps']}steps "
                  f"{cfg['h']}×{cfg['w']}", flush=True)

            with torch.inference_mode():
                output = pipe(
                    prompt_embeds=pos_gpu,
                    negative_prompt_embeds=neg_gpu,
                    num_frames=cfg["num_frames"],
                    num_inference_steps=cfg["steps"],
                    guidance_scale=5.0,
                    height=cfg["h"],
                    width=cfg["w"],
                )

            del pos_gpu, neg_gpu
            pos_gpu = neg_gpu = None
            clear_gpu()

            frames = output.frames[0]
            ok, info = check_frames_valid(frames)
            print(f"    Frames: {info}")

            if not ok:
                if attempt == 0:
                    print("    Grey output → trying smaller config...")
                    continue
                raise RuntimeError(f"Grey frames after fallback: {info}")

            export_frames(frames, clip_path, TARGET_FPS)
            sz = os.path.getsize(clip_path) if os.path.exists(clip_path) else 0
            if sz < 3000:
                raise RuntimeError(f"Output too small ({sz}B)")

            elapsed  = time.time() - t0
            clip_dur = cfg["num_frames"] / TARGET_FPS
            print(f"    ✓ {clip_dur:.1f}s | {sz//1024}KB | {elapsed:.0f}s")
            return clip_path, clip_dur

        except torch.cuda.OutOfMemoryError as e:
            print(f"    OOM: {str(e)[:90]}")
            if attempt >= len(configs) - 1:
                raise Exception(f"OOM on all configs: {str(e)[:120]}")

        finally:
            if pos_gpu is not None: del pos_gpu
            if neg_gpu is not None: del neg_gpu
            clear_gpu()

    raise Exception("All generation attempts failed")


# ── final assembly ────────────────────────────────────────────────
def stitch_clips_with_audio(record_id, clip_paths, clip_durations,
                             audio_path, audio_duration):
    final_path  = f"{VIDEO_DIR}/{record_id}.mp4"
    concat_list = f"{CLIPS_DIR}/{record_id}_concat.txt"
    concat_path = f"{CLIPS_DIR}/{record_id}_raw.mp4"

    total_dur    = sum(clip_durations)
    speed_factor = max(0.5, min(2.0, total_dur / audio_duration))
    print(f"  clips={total_dur:.1f}s  audio={audio_duration:.1f}s  speed={speed_factor:.2f}×")

    with open(concat_list, 'w') as f:
        for cp in clip_paths:
            f.write(f"file '{os.path.abspath(cp)}'\n")

    r = subprocess.run(
        ['ffmpeg','-y','-f','concat','-safe','0',
         '-i',concat_list,'-c','copy',concat_path],
        capture_output=True, text=True)
    if r.returncode != 0:
        subprocess.run(
            ['ffmpeg','-y','-f','concat','-safe','0','-i',concat_list,
             '-c:v','libx264','-crf','23',concat_path],
            capture_output=True)

    pts = f"{1.0/speed_factor:.4f}"
    vf  = (
        f"setpts={pts}*PTS,"
        f"scale=1280:720:force_original_aspect_ratio=decrease,"
        f"pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1"
    )
    r = subprocess.run([
        'ffmpeg','-y',
        '-i',concat_path,'-i',audio_path,
        '-map','0:v','-map','1:a',
        '-vf',vf,
        '-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p',
        '-c:a','aac','-b:a','128k','-ac','2','-ar','44100',
        '-t',str(audio_duration),
        '-movflags','+faststart',
        final_path
    ], capture_output=True, text=True)

    if r.returncode != 0:
        raise Exception(f"FFmpeg failed:\n{r.stderr[-400:]}")

    sz = os.path.getsize(final_path) / 1024 / 1024
    print(f"  ✓ {final_path}  {sz:.1f}MB  1280×720  {audio_duration:.1f}s\n")
    try:
        os.remove(concat_path); os.remove(concat_list)
    except Exception:
        pass
    return final_path


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def process_job(job_data):
    record_id  = job_data["recordId"]
    characters = job_data["characters"]
    emotions   = job_data["emotions"]
    narration  = job_data["narration"]
    lang       = job_data.get("language", "hi")

    print("\n" + "═"*65)
    print(f"  JOB: {record_id}")
    print("═"*65)
    for i, c in enumerate(characters):
        print(f"  char {i+1}: [{detect_character_type(c)}] {c[:60]}")
    print(f"  emotions : {' → '.join(emotions)}")
    print(f"  narration: {narration[:70]}...")
    print()
    start = time.time()

    print("━━━ STEP 1: AUDIO ━━━")
    audio_path, audio_duration = generate_audio(
        record_id, narration, characters, emotions, lang)

    print("━━━ STEP 2: SCENE PLANNING ━━━")
    scenes = split_into_scenes(narration, emotions)

    print("━━━ STEP 3: PROMPTS ━━━")
    prompts_pos, prompts_neg = [], []
    for sc in scenes:
        p, n = build_scene_prompt(characters, sc)
        prompts_pos.append(p)
        prompts_neg.append(n)
        print(f"  [{sc['emotion']:10s}] {p[:65]}...")
    print()

    print("━━━ STEP 4: TEXT ENCODING (CPU) ━━━")
    pos_embeds, neg_embeds = encode_prompts_on_cpu(prompts_pos, prompts_neg)

    n = len(scenes)
    print(f"━━━ STEP 5: VIDEO GENERATION ({n} clip{'s' if n>1 else ''}) ━━━")
    print(f"  Est: {n*4}–{n*7} min on T4\n")

    clip_paths, clip_durations = [], []
    for i, scene in enumerate(scenes):
        print(f"\n  [Clip {i+1}/{n}]  emotion={scene['emotion']}")
        cp, cd = generate_clip(record_id, i, pos_embeds[i], neg_embeds[i])
        clip_paths.append(cp)
        clip_durations.append(cd)
        elapsed = time.time() - start
        eta     = (elapsed / (i + 1)) * (n - i - 1)
        print(f"  Elapsed: {elapsed/60:.1f}m  ETA: {eta/60:.1f}m")

    print("\n━━━ STEP 6: FINAL ASSEMBLY ━━━")
    final_path = stitch_clips_with_audio(
        record_id, clip_paths, clip_durations, audio_path, audio_duration)

    for cp in clip_paths:
        try: os.remove(cp)
        except Exception: pass

    unload_video_pipeline()

    elapsed = time.time() - start
    result = {
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
        "model":          "Wan2.1-T2V-1.3B-Diffusers (v5.1)",
        "processingTime": round(elapsed, 2),
        "timestamp":      time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(f"{RESULT_DIR}/{record_id}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n✓ JOB DONE: {elapsed/60:.1f} min | {final_path}")
    return result


if __name__ == "__main__":
    if torch.cuda.is_available():
        u, t = get_vram()
        print(f"GPU  : {torch.cuda.get_device_name(0)}")
        print(f"VRAM : {t:.1f}GB total | {u:.1f}GB used")
        print(f"cuDNN: enabled={torch.backends.cudnn.enabled} "
              f"benchmark={torch.backends.cudnn.benchmark}")
        print(f"Plan : text_encoder→CPU→delete | transformer+VAE→GPU (~3.8GB)")
        print(f"VRAM : ~3.8GB weights + ~6GB activations ≈ 10GB peak")
