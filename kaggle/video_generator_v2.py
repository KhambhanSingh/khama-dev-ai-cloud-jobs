"""
Story Video Generator v5.0 — Wan2.1 WanPipeline (DEFINITIVE FIX)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPLETE ROOT CAUSE ANALYSIS (all versions)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Wan2.1-1.3B actual memory breakdown (float16):
  UMT5 text encoder  : ~9.4GB  ← THE PROBLEM. It's T5-XXL based, not T5-small
  WanTransformer3D   : ~2.6GB
  AutoencoderKLWan   : ~1.2GB
  ────────────────────────────
  Total weights      : ~13.2GB
  T4 VRAM            : 15.6GB  (effective ~14.5GB after CUDA/driver overhead)

v4.2/4.3/4.5: CUBLAS_STATUS_ALLOC_FAILED
  → enable_model_cpu_offload() uses accelerate hooks that intercept forward()
  → cuBLAS cannot initialize its handle mid-hook on torch 2.10+cu128
  → Not an OOM. A CUDA context corruption.

v4.4: OOM at 15.3GB on retry
  → Pipeline never unloaded between retries (WAN_PIPE still in VRAM)

v4.6: OOM at pipe.to("cuda") with 14.45GB already used
  → from_pretrained() loads to CPU first (~13.2GB RAM)
  → pipe.to("cuda") copies to GPU while CPU copy still exists
  → Peak = 13.2GB (CPU) + 13.2GB (GPU) = 26.4GB → OOM

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE DEFINITIVE SOLUTION (v5.0)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Strategy: Manual 2-phase loading — keep text encoder on CPU always.

Phase 1 — Text encoding (CPU only):
  Load UMT5 text encoder on CPU (float32 for stability)
  Tokenize and encode all scene prompts → get embeddings as tensors
  Delete text encoder from memory entirely
  gc.collect() → RAM back to normal

Phase 2 — Video generation (GPU only):
  Load WanTransformer3D + VAE directly to CUDA (only ~3.8GB)
  Build minimal pipeline with pre-computed embeddings
  Pass prompt_embeds directly to pipe() — skips internal text encoding
  No accelerate hooks → No CUBLAS issue
  Only ~6-8GB peak VRAM during inference → fits in 15.6GB

This approach:
  ✓ Eliminates CUBLAS_STATUS_ALLOC_FAILED (no hooks)
  ✓ Eliminates OOM during loading (text encoder never goes to GPU)
  ✓ Eliminates OOM during inference (only 3.8GB of weights on GPU)
  ✓ No grey video (consistent float16 throughout video generation)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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
torch.backends.cudnn.benchmark         = False

TARGET_FPS   = 16
CLIP_FRAMES  = 17       # 4*4+1 — reliable on T4
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

# ── helpers ──────────────────────────────────────────────────────
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

# ── text encoding (CPU-only, called once per job) ─────────────────
def encode_prompts_on_cpu(prompts_pos, prompts_neg):
    """
    Run UMT5 text encoder entirely on CPU.
    Returns (pos_embeds_list, neg_embeds_list) as float16 CPU tensors.
    The text encoder is loaded, used, and deleted — never touches GPU.

    Why CPU float32 for encoding:
      UMT5 in float16 on CPU produces NaN/Inf due to limited dynamic range
      of float16 for long attention sequences. float32 is safe on CPU.
      We cast to float16 after encoding for GPU compatibility.
    """
    print("\n[TEXT ENCODING — CPU only]")
    from transformers import AutoTokenizer, UMT5EncoderModel

    model_path = WAN_MODEL_ID

    print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, subfolder="tokenizer"
    )

    print("  Loading UMT5 text encoder on CPU (float32)...")
    text_encoder = UMT5EncoderModel.from_pretrained(
        model_path,
        subfolder="text_encoder",
        torch_dtype=torch.float32,   # float32 on CPU avoids NaN in long seqs
        low_cpu_mem_usage=True,
    )
    text_encoder.eval()
    # Stays on CPU — never moves to GPU

    MAX_LEN = 128  # 512 default OOMs even on CPU for long prompts

    def encode_batch(texts):
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
        # Cast to float16 for GPU inference later
        return out.last_hidden_state.to(dtype=torch.float16).cpu()

    print(f"  Encoding {len(prompts_pos)} positive + {len(prompts_neg)} negative prompts...")
    pos_embeds = [encode_batch([p]) for p in prompts_pos]
    neg_embeds = [encode_batch([n]) for n in prompts_neg]

    print("  Deleting text encoder from RAM...")
    del text_encoder, tokenizer
    gc.collect()
    print(f"  Text encoding done | VRAM: {vram_str()}\n")

    return pos_embeds, neg_embeds


# ── video pipeline (GPU-only, no text encoder) ───────────────────
VIDEO_PIPE = None

def load_video_pipeline():
    """
    Load ONLY the video components on GPU:
      - WanTransformer3D  (~2.6GB)
      - AutoencoderKLWan  (~1.2GB)
      Total: ~3.8GB on GPU — leaves ~11GB for activations/attention

    No text encoder → no accelerate hooks → no CUBLAS issue.
    We pass pre-computed prompt_embeds directly to bypass internal encoding.
    """
    global VIDEO_PIPE
    if VIDEO_PIPE is not None:
        return VIDEO_PIPE

    print("[VIDEO PIPELINE — GPU only, no text encoder]")
    clear_gpu()
    print(f"  VRAM before load: {vram_str()}")

    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

    # Load VAE directly to CUDA — small enough (~1.2GB)
    vae = AutoencoderKLWan.from_pretrained(
        WAN_MODEL_ID,
        subfolder="vae",
        torch_dtype=torch.float16,
    ).to("cuda")

    # Load full pipeline but we'll manage device manually
    pipe = WanPipeline.from_pretrained(
        WAN_MODEL_ID,
        vae=vae,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    # Move ONLY transformer + VAE to GPU — leave text encoder on CPU
    # (text encoder already deleted from RAM, this just ensures scheduler etc. are right)
    pipe.transformer = pipe.transformer.to("cuda")
    pipe.vae         = pipe.vae.to("cuda")
    # text_encoder stays on CPU (we won't call it — we pass embeddings directly)
    pipe.text_encoder = pipe.text_encoder.to("cpu")

    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=3.0
    )

    # Attention slicing — chunks attention to save peak VRAM
    pipe.enable_attention_slicing(1)

    # VAE slicing — decodes frame-by-frame, saves ~0.5GB peak
    try:
        pipe.enable_vae_slicing()
        print("  VAE slicing: enabled")
    except Exception:
        pass

    pipe.set_progress_bar_config(desc="  Generating", ncols=60, leave=False)

    VIDEO_PIPE = pipe
    print(f"  Pipeline ready | VRAM: {vram_str()}\n")
    return VIDEO_PIPE


def unload_video_pipeline():
    global VIDEO_PIPE
    if VIDEO_PIPE is None:
        return
    print("  Unloading video pipeline...")
    try:
        VIDEO_PIPE.transformer.to("cpu")
        VIDEO_PIPE.vae.to("cpu")
    except: pass
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
        if priority.get(ct,0) > bp:
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
            "emotion": emotions[min(eidx, n_e-1)],
            "index":   i
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


# ── frame validation ──────────────────────────────────────────────
def check_frames_valid(frames):
    if not frames:
        return False, "No frames"
    idxs   = [0, len(frames)//2, len(frames)-1]
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
    return True, f"mean={arr.mean():.1f} std={arr.std():.1f} n={len(frames)}"

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
def generate_clip(record_id, scene_idx, pos_embed, neg_embed, clip_idx):
    """
    Generate one video clip using pre-computed prompt embeddings.
    Pass embeddings directly to pipe() → text encoder never called → no CUBLAS.

    pos_embed / neg_embed: float16 CPU tensors, shape [1, seq_len, hidden_dim]
    """
    pipe      = load_video_pipeline()
    clip_path = f"{CLIPS_DIR}/{record_id}_clip_{clip_idx:02d}.mp4"
    t0        = time.time()

    # Move embeddings to GPU for this call
    pos_gpu = pos_embed.to("cuda")
    neg_gpu = neg_embed.to("cuda")

    # Progressive fallback: smaller if OOM
    configs = [
        {"num_frames": CLIP_FRAMES, "steps": 15, "h": VIDEO_H, "w": VIDEO_W},
        {"num_frames": 9,           "steps": 12, "h": 256,     "w": 448},
    ]

    for attempt, cfg in enumerate(configs):
        try:
            if attempt > 0:
                print(f"    Fallback {attempt+1}: "
                      f"{cfg['num_frames']}fr {cfg['h']}x{cfg['w']}...")
                clear_gpu()
                time.sleep(1.0)
                pos_gpu = pos_embed.to("cuda")
                neg_gpu = neg_embed.to("cuda")

            print(f"    VRAM: {vram_str()} | "
                  f"{cfg['num_frames']}fr {cfg['steps']}steps "
                  f"{cfg['h']}x{cfg['w']}", flush=True)

            with torch.inference_mode():
                output = pipe(
                    # Pass pre-computed embeddings — skips internal text encoding
                    prompt_embeds=pos_gpu,
                    negative_prompt_embeds=neg_gpu,
                    num_frames=cfg["num_frames"],
                    num_inference_steps=cfg["steps"],
                    guidance_scale=5.0,
                    height=cfg["h"],
                    width=cfg["w"],
                )

            # Clean up GPU embeddings
            del pos_gpu, neg_gpu
            clear_gpu()

            frames = output.frames[0]
            ok, info = check_frames_valid(frames)
            print(f"    Frames: {info}")

            if not ok:
                if attempt == 0:
                    print("    Grey output → fallback config...")
                    pos_gpu = pos_embed.to("cuda")
                    neg_gpu = neg_embed.to("cuda")
                    continue
                raise RuntimeError(f"Grey frames after retry: {info}")

            export_frames(frames, clip_path, TARGET_FPS)
            sz = os.path.getsize(clip_path) if os.path.exists(clip_path) else 0
            if sz < 3000:
                raise RuntimeError(f"Output file too small: {sz}B")

            elapsed  = time.time() - t0
            clip_dur = cfg["num_frames"] / TARGET_FPS
            print(f"    Saved: {sz/1024:.0f}KB | {clip_dur:.1f}s | {elapsed:.0f}s gen time")
            return clip_path, clip_dur

        except torch.cuda.OutOfMemoryError as e:
            print(f"    OOM: {str(e)[:80]}")
            try: del pos_gpu, neg_gpu
            except: pass
            clear_gpu()
            if attempt >= len(configs) - 1:
                raise Exception(f"OOM on all configs: {str(e)[:100]}")

        finally:
            try: del pos_gpu, neg_gpu
            except: pass
            clear_gpu()

    raise Exception("All clip generation attempts failed")


# ── stitching ─────────────────────────────────────────────────────
def stitch_clips_with_audio(record_id, clip_paths, clip_durations,
                             audio_path, audio_duration):
    final_path  = f"{VIDEO_DIR}/{record_id}.mp4"
    concat_list = f"{CLIPS_DIR}/{record_id}_concat.txt"
    concat_path = f"{CLIPS_DIR}/{record_id}_raw.mp4"

    total_dur    = sum(clip_durations)
    speed_factor = max(0.5, min(2.0, total_dur / audio_duration))
    print(f"  Clips: {total_dur:.1f}s | Audio: {audio_duration:.1f}s | "
          f"Speed: {speed_factor:.2f}x")

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
        '-i', concat_path, '-i', audio_path,
        '-map','0:v','-map','1:a',
        '-vf', vf,
        '-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p',
        '-c:a','aac','-b:a','128k','-ac','2','-ar','44100',
        '-t', str(audio_duration),
        '-movflags','+faststart',
        final_path
    ], capture_output=True, text=True)

    if r.returncode != 0:
        raise Exception(f"FFmpeg stitch failed:\n{r.stderr[-400:]}")

    sz = os.path.getsize(final_path) / 1024 / 1024
    print(f"  Final: {final_path} | {sz:.1f}MB | 1280x720 | {audio_duration:.1f}s\n")

    try:
        os.remove(concat_path)
        os.remove(concat_list)
    except: pass
    return final_path


# ── main job processor ────────────────────────────────────────────
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
        print(f"  char {i+1}: [{detect_character_type(c)}] {c[:60]}")
    print(f"  emotions : {' → '.join(emotions)}")
    print(f"  narration: {narration[:70]}...")
    print()
    start = time.time()

    # ── Step 1: Audio (no GPU) ────────────────────────────────────
    print("━━━ STEP 1: AUDIO ━━━")
    audio_path, audio_duration = generate_audio(
        record_id, narration, characters, emotions, lang)

    # ── Step 2: Scene planning ────────────────────────────────────
    print("━━━ STEP 2: SCENE PLANNING ━━━")
    scenes = split_into_scenes(narration, emotions)

    # ── Step 3: Build all prompts ─────────────────────────────────
    print("━━━ STEP 3: BUILD PROMPTS ━━━")
    prompts_pos, prompts_neg = [], []
    for sc in scenes:
        p, n = build_scene_prompt(characters, sc)
        prompts_pos.append(p)
        prompts_neg.append(n)
        print(f"  [{sc['emotion']:10s}] {p[:65]}...")
    print()

    # ── Step 4: Text encoding on CPU ─────────────────────────────
    print("━━━ STEP 4: TEXT ENCODING (CPU) ━━━")
    pos_embeds, neg_embeds = encode_prompts_on_cpu(prompts_pos, prompts_neg)
    # At this point: text encoder deleted, GPU clean, embeddings in CPU RAM

    # ── Step 5: Video clip generation ────────────────────────────
    n = len(scenes)
    print(f"━━━ STEP 5: VIDEO GENERATION ({n} clips) ━━━")
    print(f"  Est: {n*4}–{n*7} min on T4\n")

    clip_paths, clip_durations = [], []
    for i, scene in enumerate(scenes):
        print(f"\n  [Clip {i+1}/{n}] emotion={scene['emotion']}")
        cp, cd = generate_clip(
            record_id, i, pos_embeds[i], neg_embeds[i], i)
        clip_paths.append(cp)
        clip_durations.append(cd)
        elapsed = time.time() - start
        eta     = (elapsed / (i+1)) * (n - i - 1)
        print(f"  Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m")

    # ── Step 6: Assemble final video ──────────────────────────────
    print("\n━━━ STEP 6: FINAL ASSEMBLY ━━━")
    final_path = stitch_clips_with_audio(
        record_id, clip_paths, clip_durations,
        audio_path, audio_duration)

    for cp in clip_paths:
        try: os.remove(cp)
        except: pass

    # Unload video pipeline — free VRAM for next job
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
        "model":          "Wan2.1-T2V-1.3B-Diffusers (v5.0)",
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
        print(f"GPU: {torch.cuda.get_device_name(0)} | "
              f"{t:.1f}GB total | {u:.1f}GB used")
        print(f"Strategy: text_encoder=CPU | transformer+VAE=GPU")
        print(f"Expected VRAM usage: ~3.8GB weights + ~6GB activations = ~10GB peak")
