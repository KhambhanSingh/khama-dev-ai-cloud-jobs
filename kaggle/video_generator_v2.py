"""
🎬 Smart Video Generator v2.0
- Proper audio-video synchronization
- Character-aware voice generation
- Emotion-based animations
- Kaggle Free Tier optimized
"""

# Install dependencies
import subprocess
import sys

def install_packages():
    packages = [
        'diffusers',
        'accelerate', 
        'transformers',
        'gTTS',
        'pydub',
        'moviepy',
        'pillow'
    ]
    
    print("📦 Installing packages...")
    for pkg in packages:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])
    
    # Install ffmpeg
    subprocess.run(['apt', 'install', '-y', 'ffmpeg'], 
                   stdout=subprocess.DEVNULL, 
                   stderr=subprocess.DEVNULL)
    print("✅ Packages installed!\n")

# Uncomment below line for first run
# install_packages()

import os
import json
import time
import torch
import gc
from diffusers import DiffusionPipeline, AutoPipelineForImage2Image
from PIL import Image
from gtts import gTTS
from pydub import AudioSegment
from pydub.effects import speedup
import glob

# ==================== CONFIGURATION ====================

# Directories
BASE_DIR = "cloud-jobs"
QUEUE_DIR = f"{BASE_DIR}/queue"
FRAMES_DIR = f"{BASE_DIR}/frames"
AUDIO_DIR = f"{BASE_DIR}/audio"
VIDEO_DIR = f"{BASE_DIR}/videos"
RESULT_DIR = f"{BASE_DIR}/results"

# Create directories
for d in [QUEUE_DIR, FRAMES_DIR, AUDIO_DIR, VIDEO_DIR, RESULT_DIR]:
    os.makedirs(d, exist_ok=True)

# GPU Memory optimization
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.backends.cuda.matmul.allow_tf32 = True

# ==================== EMOTION MAPPING ====================

EMOTION_MAP = {
    "neutral": ("standing normally with calm expression", 0.10),
    "happy": ("smiling brightly with joyful expression", 0.13),
    "sad": ("shoulders drooping with downcast eyes", 0.12),
    "excited": ("leaning forward energetically with wide eyes", 0.15),
    "angry": ("frowning with tense body language", 0.14),
    "scared": ("cowering with wide fearful eyes", 0.15),
    "surprised": ("eyes wide open and body suddenly alert", 0.16),
    "thinking": ("hand on chin in contemplative pose", 0.11),
    "running": ("moving fast with stretched legs", 0.18),
    "walking": ("walking steadily with natural gait", 0.14),
    "jumping": ("leaping into air with energy", 0.17),
    "sitting": ("sitting calmly with relaxed posture", 0.10),
    "sleeping": ("sleeping deeply with eyes closed", 0.08),
    "laughing": ("laughing heartily with open mouth", 0.14),
    "crying": ("tears flowing with sad expression", 0.13),
    "dancing": ("moving rhythmically with joy", 0.16),
    "eating": ("enjoying food with content expression", 0.11),
    "talking": ("speaking with animated gestures", 0.12),
}

# ==================== CHARACTER TYPE DETECTION ====================

CHARACTER_TYPES = {
    "animal": ["rabbit", "hare", "turtle", "tortoise", "lion", "tiger", "bear", 
               "fox", "wolf", "deer", "elephant", "monkey", "cat", "dog", "mouse",
               "squirrel", "panda", "koala"],
    "bird": ["bird", "crow", "peacock", "parrot", "eagle", "sparrow", "owl",
             "pigeon", "duck", "swan"],
    "child": ["child", "kid", "baby", "toddler", "little girl", "little boy"],
    "girl": ["girl", "princess", "woman", "lady", "female"],
    "boy": ["boy", "prince", "man", "gentleman", "male"],
}

def detect_character_type(character_desc):
    """Detect character type from description"""
    desc_lower = character_desc.lower()
    
    # Priority: bird > animal > child > girl > boy > neutral
    for char_type, keywords in CHARACTER_TYPES.items():
        if any(keyword in desc_lower for keyword in keywords):
            return char_type
    
    return "neutral"

# ==================== GPU MEMORY MANAGEMENT ====================

def clear_gpu_memory():
    """Aggressively clear GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

# ==================== MODEL LOADING ====================

def load_model():
    """Load diffusion model with memory optimization"""
    print("🔧 Loading AI model (this may take 1-2 minutes)...")
    
    try:
        # Use SDXL-Turbo for fast generation
        pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
        
        pipe = pipe.to("cuda")
        
        # Enable memory optimizations
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
        pipe.set_progress_bar_config(disable=True)
        
        print("✅ Model loaded successfully!\n")
        return pipe
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        raise

# Initialize model globally
MODEL = None

def get_model():
    """Get or initialize model"""
    global MODEL
    if MODEL is None:
        MODEL = load_model()
    return MODEL

# ==================== AUDIO GENERATION ====================

def get_audio_duration(audio_path):
    """Get audio duration in seconds"""
    audio = AudioSegment.from_file(audio_path)
    return len(audio) / 1000.0

def apply_voice_effects(audio_segment, character_type, emotion):
    """Apply voice effects based on character and emotion"""
    
    # Character-based pitch adjustment
    pitch_adjustments = {
        "child": 1.25,
        "girl": 1.1,
        "boy": 0.9,
        "bird": 1.35,
        "animal": 0.95,
        "neutral": 1.0
    }
    
    pitch = pitch_adjustments.get(character_type, 1.0)
    
    if pitch != 1.0:
        audio_segment = audio_segment._spawn(
            audio_segment.raw_data,
            overrides={"frame_rate": int(audio_segment.frame_rate * pitch)}
        )
        audio_segment = audio_segment.set_frame_rate(44100)
    
    # Emotion-based speed adjustment
    if emotion in ["excited", "happy", "laughing", "running"]:
        audio_segment = speedup(audio_segment, playback_speed=1.1)
    elif emotion in ["sad", "crying", "sleeping", "tired"]:
        audio_segment = speedup(audio_segment, playback_speed=0.92)
    elif emotion in ["scared", "surprised", "shocked"]:
        audio_segment = speedup(audio_segment, playback_speed=1.15)
    
    return audio_segment

def detect_dominant_character(characters):
    """Find the main character for narration voice"""
    priority = {"child": 5, "girl": 4, "boy": 3, "bird": 2, "animal": 1, "neutral": 0}
    
    best_type = "neutral"
    best_priority = -1
    
    for char in characters:
        char_type = detect_character_type(char)
        if priority.get(char_type, 0) > best_priority:
            best_priority = priority[char_type]
            best_type = char_type
    
    return best_type

def generate_audio(record_id, text, characters, emotions, lang="hi"):
    """Generate audio with character-aware voice"""
    
    mp3_path = f"{AUDIO_DIR}/{record_id}.mp3"
    wav_path = f"{AUDIO_DIR}/{record_id}.wav"
    
    # Detect narrator character
    narrator_type = detect_dominant_character(characters)
    dominant_emotion = emotions[-1] if emotions else "neutral"
    
    print(f"🎤 Voice Profile:")
    print(f"   Narrator: {narrator_type}")
    print(f"   Emotion: {dominant_emotion}")
    
    # Generate TTS
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save(mp3_path)
    
    # Apply effects
    audio = AudioSegment.from_file(mp3_path)
    audio = apply_voice_effects(audio, narrator_type, dominant_emotion)
    
    # Export
    audio.export(wav_path, format="wav")
    
    duration = len(audio) / 1000.0
    print(f"   Duration: {duration:.2f}s")
    print(f"✅ Audio generated!\n")
    
    return wav_path, duration

# ==================== FRAME GENERATION ====================

def build_prompt(characters, emotion):
    """Build image generation prompt"""
    action, _ = EMOTION_MAP.get(emotion, ("standing normally", 0.10))
    
    # Join characters
    if len(characters) == 1:
        joined = characters[0]
    elif len(characters) == 2:
        joined = f"{characters[0]} and {characters[1]}"
    else:
        joined = ", ".join(characters[:-1]) + f", and {characters[-1]}"
    
    # Enhanced prompt for better quality
    prompt = (
        f"{joined}, {action}, "
        f"in a beautiful meadow with clear sky, "
        f"cartoon style, vibrant colors, "
        f"consistent character design, professional animation quality"
    )
    
    return prompt

def generate_frames(record_id, characters, emotion_timeline, audio_duration):
    """
    Generate frames synchronized with audio duration
    
    Args:
        record_id: Unique identifier
        characters: List of character descriptions
        emotion_timeline: List of emotions
        audio_duration: Audio duration in seconds
    
    Returns:
        total_frames: Number of frames generated
    """
    
    # Calculate frame count based on audio duration
    TARGET_FPS = 10  # Lower FPS for Kaggle free tier
    total_frames = max(int(audio_duration * TARGET_FPS), len(emotion_timeline) * 3)
    
    print(f"🎬 Frame Generation:")
    print(f"   Audio Duration: {audio_duration:.2f}s")
    print(f"   Target FPS: {TARGET_FPS}")
    print(f"   Total Frames: {total_frames}")
    print(f"   Emotions: {len(emotion_timeline)}\n")
    
    pipe = get_model()
    
    frames_per_emotion = max(1, total_frames // len(emotion_timeline))
    current_frame = 0
    image_size = 512  # Standard size for SDXL-Turbo
    
    prev_image = None
    
    for emotion_idx, emotion in enumerate(emotion_timeline):
        action, strength = EMOTION_MAP.get(emotion, ("standing normally", 0.10))
        
        frames_for_this_emotion = frames_per_emotion
        # Last emotion gets remaining frames
        if emotion_idx == len(emotion_timeline) - 1:
            frames_for_this_emotion = total_frames - current_frame
        
        print(f"🎭 Emotion {emotion_idx + 1}/{len(emotion_timeline)}: {emotion}")
        print(f"   Generating {frames_for_this_emotion} frames...")
        
        for frame_in_emotion in range(frames_for_this_emotion):
            if current_frame >= total_frames:
                break
            
            frame_path = f"{FRAMES_DIR}/{record_id}_{current_frame:04d}.png"
            prompt = build_prompt(characters, emotion)
            
            try:
                if current_frame == 0:
                    # First frame - generate from scratch
                    with torch.inference_mode():
                        result = pipe(
                            prompt=prompt,
                            height=image_size,
                            width=image_size,
                            num_inference_steps=1,
                            guidance_scale=0.0
                        )
                        image = result.images[0]
                else:
                    # Subsequent frames - use image-to-image
                    with torch.inference_mode():
                        result = pipe(
                            prompt=prompt,
                            image=prev_image,
                            strength=float(strength),
                            num_inference_steps=1,
                            guidance_scale=0.0
                        )
                        image = result.images[0]
                
                # Save frame
                image.save(frame_path)
                prev_image = image.copy()
                
                # Progress indicator
                if (current_frame + 1) % 5 == 0 or current_frame == 0:
                    print(f"   Frame {current_frame + 1}/{total_frames} ✓")
                
                current_frame += 1
                
                # Clear memory every 10 frames
                if current_frame % 10 == 0:
                    clear_gpu_memory()
                
            except Exception as e:
                print(f"   ❌ Frame {current_frame} failed: {e}")
                # Use previous frame as fallback
                if prev_image and current_frame > 0:
                    prev_frame_path = f"{FRAMES_DIR}/{record_id}_{current_frame-1:04d}.png"
                    if os.path.exists(prev_frame_path):
                        Image.open(prev_frame_path).save(frame_path)
                        current_frame += 1
                else:
                    raise
    
    print(f"✅ Generated {current_frame} frames!\n")
    
    # Clean up
    clear_gpu_memory()
    
    return current_frame

# ==================== VIDEO CREATION ====================

def create_video_from_frames(record_id, audio_path, total_frames, audio_duration):
    """
    Create video from frames and merge with audio
    
    Args:
        record_id: Unique identifier
        audio_path: Path to audio file
        total_frames: Number of frames generated
        audio_duration: Audio duration in seconds
    
    Returns:
        final_video_path: Path to final video
    """
    
    video_path = f"{VIDEO_DIR}/{record_id}.mp4"
    
    # Calculate FPS for perfect sync
    fps = total_frames / audio_duration
    fps = max(8, min(fps, 24))  # Clamp between 8-24 FPS
    
    print(f"🎬 Creating Video:")
    print(f"   Frames: {total_frames}")
    print(f"   Audio: {audio_duration:.2f}s")
    print(f"   FPS: {fps:.2f}\n")
    
    # FFmpeg command to create video with audio
    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', f'{FRAMES_DIR}/{record_id}_%04d.png',
        '-i', audio_path,
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
        '-c:a', 'aac',
        '-b:a', '192k',
        '-shortest',  # Match shortest stream (audio)
        '-movflags', '+faststart',  # Enable fast start for web playback
        video_path
    ]
    
    # Run FFmpeg
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    if result.returncode != 0:
        print(f"⚠️ FFmpeg stderr: {result.stderr}")
        raise Exception(f"Video creation failed: {result.stderr}")
    
    # Verify video was created
    if not os.path.exists(video_path):
        raise Exception("Video file was not created")
    
    video_size_mb = os.path.getsize(video_path) / (1024 * 1024)
    print(f"✅ Video created!")
    print(f"   Path: {video_path}")
    print(f"   Size: {video_size_mb:.2f} MB\n")
    
    return video_path

def cleanup_frames(record_id):
    """Delete frame files to save space"""
    frame_pattern = f"{FRAMES_DIR}/{record_id}_*.png"
    deleted = 0
    
    for frame_file in glob.glob(frame_pattern):
        try:
            os.remove(frame_file)
            deleted += 1
        except:
            pass
    
    if deleted > 0:
        print(f"🗑️  Cleaned up {deleted} frame files\n")

# ==================== MAIN JOB PROCESSOR ====================

def process_job(job_data):
    """
    Process a single video generation job
    
    Args:
        job_data: Dictionary containing job parameters
        
    Returns:
        result: Dictionary with job results
    """
    
    record_id = job_data["recordId"]
    characters = job_data["characters"]
    emotions = job_data["emotions"]
    narration = job_data["narration"]
    lang = job_data.get("language", "hi")
    
    print("\n" + "="*70)
    print(f"🎬 STARTING JOB: {record_id}")
    print("="*70)
    print(f"📝 Characters: {len(characters)}")
    for i, char in enumerate(characters):
        char_type = detect_character_type(char)
        print(f"   {i+1}. [{char_type}] {char[:60]}")
    
    print(f"\n🎭 Emotions: {' → '.join(emotions)}")
    print(f"💬 Narration: {narration[:100]}...")
    print()
    
    try:
        start_time = time.time()
        
        # Step 1: Generate Audio
        print("="*70)
        print("STEP 1: AUDIO GENERATION")
        print("="*70)
        audio_path, audio_duration = generate_audio(
            record_id, narration, characters, emotions, lang
        )
        
        # Step 2: Generate Frames
        print("="*70)
        print("STEP 2: FRAME GENERATION")
        print("="*70)
        total_frames = generate_frames(
            record_id, characters, emotions, audio_duration
        )
        
        # Step 3: Create Video
        print("="*70)
        print("STEP 3: VIDEO CREATION")
        print("="*70)
        video_path = create_video_from_frames(
            record_id, audio_path, total_frames, audio_duration
        )
        
        # Step 4: Cleanup
        cleanup_frames(record_id)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Save result
        result = {
            "recordId": record_id,
            "status": "SUCCESS",
            "video": video_path,
            "audio": audio_path,
            "characters": characters,
            "characterTypes": [detect_character_type(c) for c in characters],
            "emotions": emotions,
            "totalFrames": total_frames,
            "audioDuration": round(audio_duration, 2),
            "fps": round(total_frames / audio_duration, 2),
            "processingTime": round(processing_time, 2),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        result_path = f"{RESULT_DIR}/{record_id}.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # Success summary
        print("="*70)
        print("✅ JOB COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"📊 Final Stats:")
        print(f"   Video: {video_path}")
        print(f"   Duration: {audio_duration:.2f}s")
        print(f"   Frames: {total_frames}")
        print(f"   FPS: {result['fps']}")
        print(f"   Processing Time: {processing_time:.1f}s")
        print(f"   Characters: {', '.join(result['characterTypes'])}")
        print("="*70 + "\n")
        
        return result
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        
        print("\n" + "="*70)
        print(f"❌ JOB FAILED: {record_id}")
        print("="*70)
        print(f"Error: {str(e)}")
        print(f"\nTraceback:\n{error_trace}")
        print("="*70 + "\n")
        
        # Save error result
        result = {
            "recordId": record_id,
            "status": "FAILED",
            "error": str(e),
            "errorType": type(e).__name__,
            "traceback": error_trace,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        result_path = f"{RESULT_DIR}/{record_id}_FAILED.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # Cleanup on failure
        cleanup_frames(record_id)
        clear_gpu_memory()
        
        raise

# ==================== QUEUE PROCESSOR ====================

def process_queue():
    """Process all jobs in queue"""
    
    job_files = [f for f in os.listdir(QUEUE_DIR) if f.endswith('.json')]
    
    if not job_files:
        print("📭 No jobs in queue")
        return
    
    print(f"\n📦 Found {len(job_files)} job(s) in queue\n")
    
    for job_file in job_files:
        job_path = f"{QUEUE_DIR}/{job_file}"
        
        try:
            # Load job
            with open(job_path, "r", encoding="utf-8") as f:
                job_data = json.load(f)
            
            # Process job
            process_job(job_data)
            
            # Remove from queue
            os.remove(job_path)
            print(f"🗑️  Removed job file: {job_file}\n")
            
        except Exception as e:
            print(f"❌ Error processing {job_file}: {e}\n")
            # Don't remove failed jobs - keep for retry
            continue

# ==================== MAIN ENTRY POINT ====================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 SMART VIDEO GENERATOR v2.0")
    print("="*70)
    print("Features:")
    print("  ✨ Multi-character support (1-5 characters)")
    print("  🎤 Character-aware voice generation")
    print("  🎭 Emotion-based animations")
    print("  🎬 Perfect audio-video sync")
    print("  💾 Kaggle Free Tier optimized")
    print("="*70 + "\n")
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")
    else:
        print("⚠️  No GPU detected - will be very slow!\n")
    
    # Process queue once
    print("Starting queue processor...\n")
    process_queue()
    
    print("\n✅ All done!")
