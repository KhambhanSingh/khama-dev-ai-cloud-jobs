"""
🎬 Story Video Generator v4.0 — Wan2.1 T2V
- Uses Wan2.1-T2V-1.3B (fits T4 15.6GB VRAM)
- Narration → scenes → text2video clips → stitch + audio
- Each scene = unique animated clip (actual motion, not static image)
- YouTube ready: 1280x720, stereo, 24fps
"""

import os, sys, json, time, gc, re, glob, subprocess, shutil
import torch
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
WAN_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B"

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

WAN_PIPE = None

def get_wan_pipeline():
    global WAN_PIPE
    if WAN_PIPE is not None:
        return WAN_PIPE
    print("🔧 Loading Wan2.1-T2V-1.3B...")
    print("   First load ~3-5 min (downloading ~12GB weights)...")
    from diffusers import AutoPipelineForText2Video
    pipe = AutoPipelineForText2Video.from_pretrained(
        WAN_MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()
    pipe.set_progress_bar_config(desc="  Generating", ncols=60, leave=False)
    WAN_PIPE = pipe
    print("✅ Wan2.1 ready!\n")
    return WAN_PIPE

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
    print(f"🎤 Narrator: {narrator} | Emotion: {dom_emotion}")
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save(mp3_path)
    audio = AudioSegment.from_file(mp3_path)
    audio = apply_voice_effects(audio, narrator, dom_emotion)
    audio.export(wav_path, format="wav")
    duration = len(audio) / 1000.0
    print(f"   Duration: {duration:.2f}s")
    print("✅ Audio done!\n")
    return wav_path, duration

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
    print(f"📖 {len(scenes)} scenes:")
    for sc in scenes:
        print(f"   [{sc['emotion']:10s}] {sc['text'][:65]}...")
    print()
    return scenes

def get_scene_bg(scene_text):
    for pattern, bg in SCENE_BACKGROUNDS.items():
        if re.search(pattern, scene_text):
            return bg
    return DEFAULT_BG

def build_wan_prompt(characters, scene):
    char_desc = characters[0][:100] if characters else "cartoon crow"
    char_type = detect_character_type(char_desc)
    motion    = EMOTION_MOTION.get(scene["emotion"], "moving naturally")
    bg        = get_scene_bg(scene["text"])
    style     = "2D cartoon animation style, colorful, child-friendly, vibrant, smooth animation"
    prompt = (
        f"{char_desc}, {motion}, "
        f"{bg}, "
        f"{style}, "
        f"high quality animated video, smooth movement, professional cartoon"
    )
    neg = "blurry, low quality, static image, photo realistic, human face, text, watermark, deformed"
    return prompt, neg

def generate_clip(record_id, scene, characters, clip_idx):
    prompt, neg_prompt = build_wan_prompt(characters, scene)
    print(f"   🎬 Clip {clip_idx+1}: [{scene['emotion']}]")
    print(f"      {prompt[:90]}...")
    pipe = get_wan_pipeline()
    clip_path = f"{CLIPS_DIR}/{record_id}_clip_{clip_idx:02d}.mp4"
    t0 = time.time()
    n_frames = CLIP_FRAMES
    try:
        with torch.inference_mode():
            output = pipe(
                prompt=prompt,
                negative_prompt=neg_prompt,
                num_frames=n_frames,
                num_inference_steps=20,
                guidance_scale=5.5,
                height=480,
                width=832,
            )
    except torch.cuda.OutOfMemoryError:
        clear_gpu()
        print("      ⚠️ OOM — retrying with 25 frames...")
        n_frames = 25
        with torch.inference_mode():
            output = pipe(
                prompt=prompt,
                negative_prompt=neg_prompt,
                num_frames=n_frames,
                num_inference_steps=15,
                guidance_scale=5.0,
                height=480,
                width=832,
            )

    frames = output.frames[0]
    temp_dir = f"{CLIPS_DIR}/tmp_{record_id}_{clip_idx}"
    os.makedirs(temp_dir, exist_ok=True)
    for fi, frame in enumerate(frames):
        frame.save(f"{temp_dir}/frame_{fi:04d}.png")

    cmd = ['ffmpeg','-y','-framerate',str(TARGET_FPS),
           '-i',f'{temp_dir}/frame_%04d.png',
           '-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p',
           clip_path]
    subprocess.run(cmd, capture_output=True)
    shutil.rmtree(temp_dir, ignore_errors=True)
    clear_gpu()

    clip_dur = len(frames) / TARGET_FPS
    size_mb  = os.path.getsize(clip_path) / 1024 / 1024 if os.path.exists(clip_path) else 0
    elapsed  = time.time() - t0
    print(f"      ✅ {clip_dur:.1f}s | {size_mb:.1f}MB | {elapsed:.0f}s gen time")
    return clip_path, clip_dur

def stitch_clips_with_audio(record_id, clip_paths, clip_durations, audio_path, audio_duration):
    final_path   = f"{VIDEO_DIR}/{record_id}.mp4"
    concat_list  = f"{CLIPS_DIR}/{record_id}_concat.txt"
    concat_path  = f"{CLIPS_DIR}/{record_id}_raw.mp4"

    total_dur    = sum(clip_durations)
    speed_factor = max(0.5, min(2.0, total_dur / audio_duration))

    print(f"\n🎞️  Stitching {len(clip_paths)} clips:")
    print(f"   Clip total: {total_dur:.1f}s | Audio: {audio_duration:.1f}s | Speed: {speed_factor:.2f}x")

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
    vf  = f"setpts={pts}*PTS,scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1"

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
        raise Exception(f"Stitch failed:\n{r.stderr[-400:]}")

    size_mb = os.path.getsize(final_path) / 1024 / 1024
    print(f"✅ Final: {final_path} | {size_mb:.1f}MB | {audio_duration:.1f}s | 1280x720\n")

    try:
        os.remove(concat_path)
        os.remove(concat_list)
    except: pass
    return final_path

def process_job(job_data):
    record_id  = job_data["recordId"]
    characters = job_data["characters"]
    emotions   = job_data["emotions"]
    narration  = job_data["narration"]
    lang       = job_data.get("language", "hi")
    job_type   = job_data.get("type", "video")

    print("\n" + "="*70)
    print(f"🎬 WAN2.1 JOB: {record_id}")
    print("="*70)
    for i,c in enumerate(characters):
        print(f"   char {i+1}: [{detect_character_type(c)}] {c[:70]}")
    print(f"   emotions: {' → '.join(emotions)}")
    print(f"   narration: {narration[:80]}...")
    print()
    start = time.time()

    print("="*50 + "\nSTEP 1: AUDIO\n" + "="*50)
    audio_path, audio_duration = generate_audio(record_id, narration, characters, emotions, lang)

    print("="*50 + "\nSTEP 2: SCENE PLANNING\n" + "="*50)
    scenes = split_into_scenes(narration, emotions)

    print("="*50 + "\nSTEP 3: WAN2.1 VIDEO CLIPS\n" + "="*50)
    print(f"   {len(scenes)} clips × ~{CLIP_FRAMES/TARGET_FPS:.1f}s each")
    print(f"   Est: {len(scenes)*12}-{len(scenes)*18} min total\n")

    clip_paths, clip_durations = [], []
    for i, scene in enumerate(scenes):
        print(f"\n── Scene {i+1}/{len(scenes)} ──")
        cp, cd = generate_clip(record_id, scene, characters, i)
        clip_paths.append(cp)
        clip_durations.append(cd)
        so_far = time.time() - start
        eta    = (so_far/(i+1)) * (len(scenes)-i-1)
        print(f"      Elapsed: {so_far/60:.1f}min | ETA: {eta/60:.1f}min")

    print("\n" + "="*50 + "\nSTEP 4: FINAL ASSEMBLY\n" + "="*50)
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
        "resolution": "1280x720", "model": "Wan2.1-T2V-1.3B",
        "processingTime": round(elapsed, 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(f"{RESULT_DIR}/{record_id}.json","w",encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("="*70)
    print(f"✅ DONE! {elapsed/60:.1f} min | {final_path}")
    print("="*70 + "\n")
    return result

if __name__ == "__main__":
    print("🎬 Wan2.1 Video Generator v4.0")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
