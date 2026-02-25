"""
🎬 Smart Video Generator v3.0 — Story-Accurate Scene Animation
- Narration split into scenes (1 sentence = 1 scene)
- Character master image created ONCE, reused for consistency
- Each scene gets unique background/action via img2img from master
- Smooth cross-fade transitions between scenes
- YouTube ready: 1280x720, stereo, 24fps
"""

import os, sys, json, time, gc, re, glob, subprocess
import torch
from PIL import Image, ImageFilter
from gtts import gTTS
from pydub import AudioSegment
from pydub.effects import speedup
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image

# ==================== DIRECTORIES ====================
BASE_DIR   = "cloud-jobs"
FRAMES_DIR = f"{BASE_DIR}/frames"
AUDIO_DIR  = f"{BASE_DIR}/audio"
VIDEO_DIR  = f"{BASE_DIR}/videos"
RESULT_DIR = f"{BASE_DIR}/results"

for d in [FRAMES_DIR, AUDIO_DIR, VIDEO_DIR, RESULT_DIR]:
    os.makedirs(d, exist_ok=True)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.backends.cuda.matmul.allow_tf32 = True

TARGET_FPS = 24  # YouTube standard

# ==================== SCENE BACKGROUNDS ====================
# Story-relevant backgrounds mapped to keywords in narration
SCENE_BACKGROUNDS = {
    "pyaas|pyasa|thirsty|paani|water|talash|searching": "vast open landscape, dry terrain, scorching sun, no water visible",
    "bageeche|bageecha|garden|phool|flowers|ped|tree":  "lush green garden, flowers blooming, trees, peaceful",
    "ghada|ghara|pot|matka|pitcher":                    "clay water pot on ground, garden background, sunlight",
    "patthar|stone|kankar|giraya|drop":                 "stones pebbles on ground, water pot nearby",
    "paani upar|water rising|bhar|filling":             "water level rising in clay pot, stones inside",
    "khushi|happy|prasann|excited|enjoy":               "bright sunny day, joyful atmosphere, colorful flowers",
    "piya|peena|drink|drinking":                        "close to water pot drinking water, satisfied",
    "soch|thinking|idea|socha|samjha":                  "thoughtful moment, light bulb idea, contemplating",
    "uda|flying|ud|soar|urana":                         "open sky, clouds, flying through air",
    "thaka|tired|baith|sitting|rest":                   "resting on tree branch, peaceful shade",
}

DEFAULT_BACKGROUND = "beautiful meadow, clear blue sky, soft sunlight, peaceful nature"

# ==================== EMOTION MAPPING ====================
EMOTION_PROMPTS = {
    "sad":      ("drooping wings, downcast eyes, slumped posture, dejected", 0.40),
    "excited":  ("wings spread wide, bright eyes, alert posture, energetic", 0.38),
    "thinking": ("head tilted, one wing raised to chin, contemplative gaze", 0.35),
    "running":  ("wings flapping fast, body leaning forward, dynamic motion", 0.42),
    "happy":    ("wings slightly open, bright cheerful eyes, upright proud posture", 0.36),
    "relaxed":  ("wings folded neatly, calm eyes, comfortable perched position", 0.30),
    "angry":    ("ruffled feathers, sharp gaze, tense body", 0.40),
    "scared":   ("wings pulled in tight, wide fearful eyes, cowering", 0.42),
    "neutral":  ("standing naturally, calm expression, wings folded", 0.28),
}

# ==================== CHARACTER TYPES ====================
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

# ==================== GPU MEMORY ====================
def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# ==================== MODEL LOADING ====================
# Two pipelines sharing weights — no extra VRAM cost
MODEL_T2I = None   # text → image  (master frame)
MODEL_I2I = None   # image → image (scene variations)

def get_models():
    global MODEL_T2I, MODEL_I2I
    if MODEL_T2I is None:
        print("🔧 Loading SDXL-Turbo text2img...")
        MODEL_T2I = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        ).to("cuda")
        MODEL_T2I.enable_attention_slicing()
        MODEL_T2I.enable_vae_slicing()
        MODEL_T2I.enable_vae_tiling()
        MODEL_T2I.set_progress_bar_config(disable=True)
        print("🔧 Loading img2img pipeline (shared weights)...")
        MODEL_I2I = AutoPipelineForImage2Image.from_pipe(MODEL_T2I)
        MODEL_I2I.set_progress_bar_config(disable=True)
        print("✅ Both pipelines ready!\n")
    return MODEL_T2I, MODEL_I2I

# ==================== AUDIO ====================
def detect_dominant_character(characters):
    priority = {"child":5,"girl":4,"boy":3,"bird":2,"animal":1,"neutral":0}
    best, best_p = "neutral", -1
    for c in characters:
        ct = detect_character_type(c)
        if priority.get(ct, 0) > best_p:
            best_p = priority[ct]
            best = ct
    return best

def apply_voice_effects(audio, char_type, emotion):
    pitch = {"child":1.25,"girl":1.1,"boy":0.9,"bird":1.35,"animal":0.95,"neutral":1.0}.get(char_type, 1.0)
    if pitch != 1.0:
        audio = audio._spawn(audio.raw_data,
                             overrides={"frame_rate": int(audio.frame_rate * pitch)})
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
    dominant_emotion = emotions[-1] if emotions else "neutral"
    print(f"🎤 Narrator: {narrator} | Emotion: {dominant_emotion}")
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save(mp3_path)
    audio = AudioSegment.from_file(mp3_path)
    audio = apply_voice_effects(audio, narrator, dominant_emotion)
    audio.export(wav_path, format="wav")
    duration = len(audio) / 1000.0
    print(f"   Duration: {duration:.2f}s")
    print("✅ Audio done!\n")
    return wav_path, duration

# ==================== SCENE PARSING ====================
def split_narration_into_scenes(narration, emotions):
    """Split narration into sentences — each becomes a scene"""
    # Split on Hindi/English sentence endings
    sentences = re.split(r'[।|\.|\!|\?]+', narration)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]

    if not sentences:
        sentences = [narration]

    # Map emotions to sentences
    scenes = []
    n_emotions = len(emotions)
    n_sentences = len(sentences)

    for i, sentence in enumerate(sentences):
        # Distribute emotions across sentences
        emotion_idx = int(i * n_emotions / n_sentences)
        emotion = emotions[min(emotion_idx, n_emotions - 1)]
        scenes.append({"text": sentence, "emotion": emotion, "index": i})

    print(f"📖 Story split into {len(scenes)} scenes:")
    for i, sc in enumerate(scenes):
        print(f"   Scene {i+1}: [{sc['emotion']}] {sc['text'][:60]}...")
    print()
    return scenes

def get_scene_background(scene_text):
    """Detect background from scene text"""
    text_lower = scene_text.lower()
    for pattern, background in SCENE_BACKGROUNDS.items():
        if re.search(pattern, text_lower):
            return background
    return DEFAULT_BACKGROUND

def build_master_prompt(characters):
    """Build the ONE consistent character description used for ALL scenes"""
    char = characters[0] if characters else "cartoon bird"
    char_type = detect_character_type(char)

    # Extract key descriptors from character description
    desc = char[:120]

    if char_type == "bird":
        style = "cartoon style, consistent character design, vibrant colors, professional 2D animation"
        base = f"{desc}, standing on ground, wings folded neatly, neutral calm expression, {style}"
    elif char_type == "animal":
        style = "cartoon style, consistent character design, vibrant colors, professional 2D animation"
        base = f"{desc}, standing naturally, neutral expression, {style}"
    elif char_type == "child":
        style = "cartoon style, child-friendly illustration, vibrant colors, professional animation"
        base = f"{desc}, standing naturally, neutral expression, {style}"
    else:
        style = "cartoon style, vibrant colors, professional 2D animation quality"
        base = f"{desc}, standing naturally, neutral expression, {style}"

    return base

def build_scene_prompt(characters, emotion, scene_text, master_prompt):
    """Build prompt for a specific scene using emotion + background context"""
    emotion_desc, _ = EMOTION_PROMPTS.get(emotion, ("standing naturally", 0.35))
    background = get_scene_background(scene_text)
    char_type = detect_character_type(characters[0] if characters else "")

    # Keep character consistent by referencing master description
    char_core = characters[0][:80] if characters else "cartoon character"

    prompt = (
        f"{char_core}, {emotion_desc}, "
        f"{background}, "
        f"cartoon animation style, vibrant colors, professional quality, "
        f"consistent character design, 2D animated film style, "
        f"high detail, clear scene"
    )
    return prompt

# ==================== IMAGE GENERATION ====================
def generate_master_image(record_id, characters, pipe_t2i=None):
    """Generate ONE master character image — used as base for ALL scenes"""
    print("🎨 Generating master character image...")
    if pipe_t2i is None:
        pipe_t2i, _ = get_models()
    master_prompt = build_master_prompt(characters)

    print(f"   Prompt: {master_prompt[:100]}...")

    with torch.inference_mode():
        result = pipe_t2i(
            prompt=master_prompt,
            negative_prompt="blurry, low quality, deformed, ugly, multiple characters, bad anatomy",
            height=768,
            width=768,
            num_inference_steps=4,
            guidance_scale=0.0
        )
    master_img = result.images[0]
    master_path = f"{FRAMES_DIR}/{record_id}_master.png"
    master_img.save(master_path)
    print(f"   ✅ Master image saved: {master_path}\n")
    clear_gpu()
    return master_img, master_path

def generate_scene_image(pipe, master_img, characters, scene, master_prompt):
    """Generate one scene image from master image using img2img"""
    emotion = scene["emotion"]
    scene_text = scene["text"]
    _, strength = EMOTION_PROMPTS.get(emotion, ("standing", 0.35))

    scene_prompt = build_scene_prompt(characters, emotion, scene_text, master_prompt)

    with torch.inference_mode():
        result = pipe(
            prompt=scene_prompt,
            image=master_img,
            strength=strength,       # Lower = more like master (consistent character)
            num_inference_steps=4,
            guidance_scale=0.0
        )
    return result.images[0]

def create_crossfade_frames(img_a, img_b, n_frames):
    """Create smooth cross-fade transition frames between two images"""
    frames = []
    for i in range(n_frames):
        alpha = i / max(n_frames - 1, 1)
        # PIL blend: 0.0 = img_a, 1.0 = img_b
        blended = Image.blend(img_a, img_b, alpha)
        frames.append(blended)
    return frames

def scale_to_youtube(img):
    """Scale image to 1280x720 with black letterbox"""
    target_w, target_h = 1280, 720
    img_ratio = img.width / img.height
    target_ratio = target_w / target_h

    if img_ratio > target_ratio:
        new_w = target_w
        new_h = int(target_w / img_ratio)
    else:
        new_h = target_h
        new_w = int(target_h * img_ratio)

    img_resized = img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    canvas.paste(img_resized, (paste_x, paste_y))
    return canvas

# ==================== MAIN FRAME GENERATION ====================
def generate_frames(record_id, characters, scenes, audio_duration):
    """
    Generate frames scene by scene:
    - Master image once
    - Per-scene unique image (consistent character, different scene/emotion)
    - Cross-fade transitions between scenes
    - Hold frames to fill audio duration
    """
    n_scenes = len(scenes)
    total_frames_needed = int(audio_duration * TARGET_FPS)

    # Allocate frames per scene
    # Give more frames to middle scenes (story action), less to transitions
    TRANSITION_FRAMES = 8   # ~0.3s cross-fade between scenes
    hold_frames_per_scene = max(
        int((total_frames_needed - (n_scenes - 1) * TRANSITION_FRAMES) / n_scenes),
        int(TARGET_FPS * 1.5)  # Minimum 1.5s per scene
    )

    print(f"🎬 Frame Plan:")
    print(f"   Total frames needed: {total_frames_needed}")
    print(f"   Scenes: {n_scenes}")
    print(f"   Hold frames/scene: {hold_frames_per_scene} ({hold_frames_per_scene/TARGET_FPS:.1f}s)")
    print(f"   Transition frames: {TRANSITION_FRAMES} per transition\n")

    pipe_t2i, pipe_i2i = get_models()
    master_prompt = build_master_prompt(characters)

    # Step 1: Generate master image
    master_img, master_path = generate_master_image(record_id, characters, pipe_t2i)

    # Step 2: Generate one image per scene
    print("🖼️  Generating scene images...")
    scene_images = []
    for i, scene in enumerate(scenes):
        print(f"   Scene {i+1}/{n_scenes}: [{scene['emotion']}] {scene['text'][:50]}...")
        scene_img = generate_scene_image(pipe_i2i, master_img, characters, scene, master_prompt)
        # Scale to YouTube dimensions immediately
        scene_img_yt = scale_to_youtube(scene_img)
        scene_images.append(scene_img_yt)

        # Save preview
        scene_img_yt.save(f"{FRAMES_DIR}/{record_id}_scene_{i:02d}.png")
        clear_gpu()
        print(f"   ✅ Scene {i+1} generated")

    print(f"\n✅ All {n_scenes} scene images generated!\n")

    # Step 3: Assemble final frame sequence
    print("🎞️  Assembling frame sequence...")
    frame_idx = 0
    master_yt = scale_to_youtube(master_img)

    for i, scene_img in enumerate(scene_images):
        # Hold frames for this scene
        for _ in range(hold_frames_per_scene):
            frame_path = f"{FRAMES_DIR}/{record_id}_{frame_idx:05d}.png"
            scene_img.save(frame_path)
            frame_idx += 1

        # Cross-fade transition to next scene (except last)
        if i < len(scene_images) - 1:
            next_img = scene_images[i + 1]
            transition_frames = create_crossfade_frames(scene_img, next_img, TRANSITION_FRAMES)
            for tf in transition_frames:
                frame_path = f"{FRAMES_DIR}/{record_id}_{frame_idx:05d}.png"
                tf.save(frame_path)
                frame_idx += 1

        if (i + 1) % 3 == 0 or i == len(scene_images) - 1:
            print(f"   Assembled {i+1}/{n_scenes} scenes → {frame_idx} frames so far")

    # Fill remaining frames if needed (repeat last scene)
    while frame_idx < total_frames_needed:
        last_frame_src = f"{FRAMES_DIR}/{record_id}_{frame_idx-1:05d}.png"
        frame_path = f"{FRAMES_DIR}/{record_id}_{frame_idx:05d}.png"
        if os.path.exists(last_frame_src):
            import shutil
            shutil.copy2(last_frame_src, frame_path)
        else:
            scene_images[-1].save(frame_path)
        frame_idx += 1

    print(f"\n✅ Total frames assembled: {frame_idx}")
    return frame_idx

# ==================== VIDEO CREATION ====================
def create_video(record_id, audio_path, total_frames, audio_duration):
    video_path = f"{VIDEO_DIR}/{record_id}.mp4"

    fps = total_frames / audio_duration
    fps = max(20, min(fps, 30))

    print(f"🎬 Creating video:")
    print(f"   Frames: {total_frames}")
    print(f"   FPS: {fps:.2f}")
    print(f"   Duration: {audio_duration:.2f}s")
    print(f"   Output: 1280x720 YouTube ready")

    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', f'{FRAMES_DIR}/{record_id}_%05d.png',
        '-i', audio_path,
        '-map', '0:v',
        '-map', '1:a',
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '18',
        '-pix_fmt', 'yuv420p',
        '-c:a', 'aac',
        '-b:a', '192k',
        '-ac', '2',
        '-ar', '44100',
        '-t', str(audio_duration),
        '-movflags', '+faststart',
        video_path
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        print(f"❌ FFmpeg error:\n{result.stderr[-500:]}")
        raise Exception(f"FFmpeg failed: {result.stderr[-200:]}")

    size_mb = os.path.getsize(video_path) / 1024 / 1024
    print(f"✅ Video created! Size: {size_mb:.1f} MB\n")
    return video_path

def cleanup_frames(record_id):
    deleted = 0
    for f in glob.glob(f"{FRAMES_DIR}/{record_id}_*.png"):
        try:
            os.remove(f)
            deleted += 1
        except:
            pass
    if deleted:
        print(f"🗑️  Cleaned {deleted} frame files")

# ==================== MAIN JOB PROCESSOR ====================
def process_job(job_data):
    record_id  = job_data["recordId"]
    characters = job_data["characters"]
    emotions   = job_data["emotions"]
    narration  = job_data["narration"]
    lang       = job_data.get("language", "hi")
    job_type   = job_data.get("type", "video")

    print("\n" + "="*70)
    print(f"🎬 JOB: {record_id}  (type={job_type})")
    print("="*70)
    print(f"📝 Characters: {len(characters)}")
    for i, c in enumerate(characters):
        print(f"   {i+1}. [{detect_character_type(c)}] {c[:70]}")
    print(f"🎭 Emotions: {' → '.join(emotions)}")
    print(f"💬 Narration ({len(narration)} chars): {narration[:80]}...")
    print()

    start_time = time.time()

    # ── Step 1: Audio ──
    print("="*50 + "\nSTEP 1: AUDIO\n" + "="*50)
    audio_path, audio_duration = generate_audio(record_id, narration, characters, emotions, lang)

    # ── Step 2: Split narration into scenes ──
    print("="*50 + "\nSTEP 2: SCENE PLANNING\n" + "="*50)
    scenes = split_narration_into_scenes(narration, emotions)

    # ── Step 3: Generate frames ──
    print("="*50 + "\nSTEP 3: FRAME GENERATION\n" + "="*50)
    total_frames = generate_frames(record_id, characters, scenes, audio_duration)

    # ── Step 4: Create video ──
    print("="*50 + "\nSTEP 4: VIDEO CREATION\n" + "="*50)
    video_path = create_video(record_id, audio_path, total_frames, audio_duration)

    # ── Step 5: Cleanup frames ──
    cleanup_frames(record_id)

    elapsed = time.time() - start_time

    result = {
        "recordId":       record_id,
        "status":         "SUCCESS",
        "video":          video_path,
        "audio":          audio_path,
        "characters":     characters,
        "characterTypes": [detect_character_type(c) for c in characters],
        "emotions":       emotions,
        "scenes":         len(scenes),
        "totalFrames":    total_frames,
        "audioDuration":  round(audio_duration, 2),
        "fps":            round(total_frames / audio_duration, 2),
        "resolution":     "1280x720",
        "processingTime": round(elapsed, 2),
        "timestamp":      time.strftime("%Y-%m-%d %H:%M:%S")
    }

    result_path = f"{RESULT_DIR}/{record_id}.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("="*70)
    print("✅ JOB COMPLETE!")
    print(f"   Video:    {video_path}")
    print(f"   Duration: {audio_duration:.1f}s  |  Scenes: {len(scenes)}  |  Frames: {total_frames}")
    print(f"   FPS: {result['fps']}  |  Resolution: 1280x720")
    print(f"   Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print("="*70 + "\n")
    return result

if __name__ == "__main__":
    print("🎬 Video Generator v3.0 — Story Scene Mode")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    else:
        print("⚠️ No GPU!")
