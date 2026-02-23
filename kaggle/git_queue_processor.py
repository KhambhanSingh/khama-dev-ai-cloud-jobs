"""
🔄 Automated Git Queue Processor for Video Generator v2.0

Features:
- Auto pulls jobs from Git queue
- Processes videos
- Auto pushes results back to Git
- Clears processed queue
- Continuous monitoring
- Local backup option
"""

import os
import json
import time
import shutil
import subprocess
from datetime import datetime

# ==================== CONFIGURATION ====================

# Git Configuration
GIT_REPO_URL = "https://github.com/USERNAME/REPO_NAME.git"  # UPDATE THIS
GIT_BRANCH = "main"
GIT_USER_EMAIL = "your-email@example.com"  # UPDATE THIS
GIT_USER_NAME = "Your Name"  # UPDATE THIS
GIT_TOKEN = None  # Will be set from secrets or environment

# Directories
BASE_DIR = "cloud-jobs"
QUEUE_DIR = f"{BASE_DIR}/queue"
IMAGES_DIR = f"{BASE_DIR}/images"
AUDIO_DIR = f"{BASE_DIR}/audio"
VIDEO_DIR = f"{BASE_DIR}/video"
RESULT_DIR = f"{BASE_DIR}/result"
LOCAL_BACKUP_DIR = f"{BASE_DIR}/local_backup"

# Create all directories
for d in [QUEUE_DIR, IMAGES_DIR, AUDIO_DIR, VIDEO_DIR, RESULT_DIR, LOCAL_BACKUP_DIR]:
    os.makedirs(d, exist_ok=True)

# Polling interval (seconds)
POLL_INTERVAL = 10  # Check for new jobs every 10 seconds

# ==================== GIT FUNCTIONS ====================

def setup_git():
    """Setup Git configuration"""
    print("🔧 Setting up Git...")
    
    # Try to get token from Kaggle secrets
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        
        global GIT_TOKEN, GIT_USER_EMAIL, GIT_USER_NAME
        
        # Get secrets
        GIT_TOKEN = user_secrets.get_secret("GITHUB_TOKEN")
        GIT_USER_EMAIL = user_secrets.get_secret("GIT_EMAIL")
        GIT_USER_NAME = user_secrets.get_secret("GIT_NAME")
        
        print("✅ Loaded secrets from Kaggle")
    except:
        # Fallback to environment variables
        GIT_TOKEN = os.getenv("GITHUB_TOKEN")
        print("⚠️  Using environment variables for Git credentials")
    
    if not GIT_TOKEN:
        print("❌ No Git token found! Set GITHUB_TOKEN in Kaggle secrets or environment")
        return False
    
    # Configure Git
    subprocess.run(['git', 'config', '--global', 'user.email', GIT_USER_EMAIL], 
                   capture_output=True)
    subprocess.run(['git', 'config', '--global', 'user.name', GIT_USER_NAME], 
                   capture_output=True)
    
    # Set remote URL with token
    repo_with_token = GIT_REPO_URL.replace('https://', f'https://{GIT_TOKEN}@')
    subprocess.run(['git', 'remote', 'set-url', 'origin', repo_with_token], 
                   capture_output=True)
    
    print(f"✅ Git configured for {GIT_USER_NAME}")
    return True

def git_pull():
    """Pull latest changes from Git"""
    print("\n🔄 Pulling from Git...")
    
    try:
        # Reset any local changes
        subprocess.run(['git', 'reset', '--hard'], 
                       capture_output=True, check=True)
        
        # Pull latest
        result = subprocess.run(['git', 'pull', 'origin', GIT_BRANCH], 
                               capture_output=True, text=True, check=True)
        
        # Recreate directories if needed
        for d in [QUEUE_DIR, IMAGES_DIR, AUDIO_DIR, VIDEO_DIR, RESULT_DIR]:
            os.makedirs(d, exist_ok=True)
        
        print("✅ Pulled successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Git pull failed: {e.stderr}")
        return False

def git_push(message):
    """Push changes to Git"""
    print(f"\n📤 Pushing to Git: {message}")
    
    try:
        # Add all changes in cloud-jobs directory
        subprocess.run(['git', 'add', BASE_DIR], 
                       capture_output=True, check=True)
        
        # Commit
        result = subprocess.run(['git', 'commit', '-m', message], 
                               capture_output=True, text=True)
        
        if result.returncode != 0 and "nothing to commit" in result.stdout:
            print("ℹ️  No changes to commit")
            return True
        
        # Push
        subprocess.run(['git', 'push', 'origin', GIT_BRANCH], 
                       capture_output=True, check=True)
        
        print("✅ Pushed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Git push failed: {e.stderr}")
        return False

def git_init_clone():
    """Initialize or clone repository"""
    if os.path.exists('.git'):
        print("📁 Repository already exists")
        return setup_git()
    
    print(f"📥 Cloning repository from {GIT_REPO_URL}")
    
    try:
        # Clone repository
        subprocess.run(['git', 'clone', GIT_REPO_URL, '.'], 
                       capture_output=True, check=True)
        
        return setup_git()
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Git clone failed: {e.stderr}")
        return False

# ==================== BACKUP FUNCTIONS ====================

def backup_to_local(result):
    """Backup video and result to local directory"""
    record_id = result['recordId']
    
    print(f"💾 Creating local backup for {record_id}...")
    
    # Create backup folder for this job
    backup_folder = f"{LOCAL_BACKUP_DIR}/{record_id}"
    os.makedirs(backup_folder, exist_ok=True)
    
    try:
        # Copy video
        if os.path.exists(result['video']):
            shutil.copy2(result['video'], f"{backup_folder}/{record_id}.mp4")
        
        # Copy audio
        if os.path.exists(result['audio']):
            shutil.copy2(result['audio'], f"{backup_folder}/{record_id}.wav")
        
        # Copy result JSON
        result_json_path = f"{RESULT_DIR}/{record_id}.json"
        if os.path.exists(result_json_path):
            shutil.copy2(result_json_path, f"{backup_folder}/result.json")
        
        print(f"✅ Backed up to {backup_folder}")
        return True
        
    except Exception as e:
        print(f"⚠️  Backup failed: {e}")
        return False

def download_result(result):
    """
    Create a download-ready package
    (In Kaggle, files in output will be downloadable)
    """
    record_id = result['recordId']
    
    # Create download directory
    download_dir = f"downloads/{record_id}"
    os.makedirs(download_dir, exist_ok=True)
    
    try:
        # Copy files
        shutil.copy2(result['video'], f"{download_dir}/{record_id}.mp4")
        shutil.copy2(result['audio'], f"{download_dir}/{record_id}.wav")
        
        # Create info file
        info = {
            "recordId": record_id,
            "video": f"{record_id}.mp4",
            "audio": f"{record_id}.wav",
            "duration": result['audioDuration'],
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(f"{download_dir}/info.json", "w") as f:
            json.dump(info, f, indent=2)
        
        print(f"📥 Download package ready: {download_dir}")
        return download_dir
        
    except Exception as e:
        print(f"⚠️  Download package creation failed: {e}")
        return None

# ==================== QUEUE PROCESSING ====================

def get_pending_jobs():
    """Get list of pending jobs from queue"""
    if not os.path.exists(QUEUE_DIR):
        return []
    
    jobs = [f for f in os.listdir(QUEUE_DIR) if f.endswith('.json')]
    return sorted(jobs)  # Process in order

def clear_processed_job(job_filename):
    """Remove processed job from queue"""
    job_path = f"{QUEUE_DIR}/{job_filename}"
    
    try:
        if os.path.exists(job_path):
            os.remove(job_path)
            print(f"🗑️  Cleared job: {job_filename}")
            return True
    except Exception as e:
        print(f"⚠️  Failed to clear job: {e}")
    
    return False

def cleanup_temp_files(record_id):
    """Clean up temporary files after processing"""
    import glob
    
    # Remove frame images
    for frame_file in glob.glob(f"{IMAGES_DIR}/{record_id}_*.png"):
        try:
            os.remove(frame_file)
        except:
            pass
    
    print(f"🧹 Cleaned up temporary files for {record_id}")

# ==================== MAIN PROCESSING LOOP ====================

def process_queue_with_git():
    """
    Main processing function:
    1. Pull from Git
    2. Process jobs
    3. Push results back
    4. Clear queue
    """
    
    # Import video generator
    try:
        import video_generator_v2 as vg
        print("✅ Video generator loaded\n")
    except ImportError:
        print("❌ video_generator_v2.py not found!")
        print("   Please ensure video_generator_v2.py is in the same directory")
        return False
    
    # Pull latest changes
    if not git_pull():
        print("⚠️  Git pull failed, continuing with local queue...")
    
    # Get pending jobs
    pending_jobs = get_pending_jobs()
    
    if not pending_jobs:
        print("📭 No jobs in queue")
        return True
    
    print(f"\n📦 Found {len(pending_jobs)} job(s) in queue")
    print("="*70 + "\n")
    
    processed_count = 0
    failed_count = 0
    
    for job_file in pending_jobs:
        job_path = f"{QUEUE_DIR}/{job_file}"
        
        try:
            # Load job
            print(f"📄 Processing: {job_file}")
            with open(job_path, 'r', encoding='utf-8') as f:
                job_data = json.load(f)
            
            # Process video
            result = vg.process_job(job_data)
            
            # Backup locally
            backup_to_local(result)
            
            # Create download package
            download_result(result)
            
            # Clean up temporary files
            cleanup_temp_files(result['recordId'])
            
            # Clear job from queue
            clear_processed_job(job_file)
            
            processed_count += 1
            
            print(f"✅ {job_file} completed successfully\n")
            
        except Exception as e:
            print(f"❌ {job_file} failed: {e}\n")
            failed_count += 1
            
            # Keep failed job in queue for manual review
            # Or optionally move to failed folder
            failed_dir = f"{BASE_DIR}/failed"
            os.makedirs(failed_dir, exist_ok=True)
            
            try:
                shutil.move(job_path, f"{failed_dir}/{job_file}")
                print(f"📁 Moved to failed: {job_file}\n")
            except:
                pass
    
    # Summary
    print("="*70)
    print(f"📊 Processing Summary:")
    print(f"   ✅ Successful: {processed_count}")
    print(f"   ❌ Failed: {failed_count}")
    print(f"   📦 Total: {len(pending_jobs)}")
    print("="*70 + "\n")
    
    # Push results back to Git
    if processed_count > 0:
        commit_msg = f"🎬 Processed {processed_count} video(s) - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        git_push(commit_msg)
    
    return True

# ==================== CONTINUOUS MONITORING ====================

def run_continuous_monitoring(check_interval=POLL_INTERVAL):
    """
    Continuously monitor queue and process jobs
    """
    
    print("\n" + "="*70)
    print("🚀 AUTOMATED GIT QUEUE PROCESSOR")
    print("="*70)
    print(f"📂 Repository: {GIT_REPO_URL}")
    print(f"🔄 Polling interval: {check_interval}s")
    print(f"📁 Queue directory: {QUEUE_DIR}")
    print(f"📁 Results directory: {RESULT_DIR}")
    print("="*70 + "\n")
    
    # Initialize Git
    if not git_init_clone():
        print("❌ Git initialization failed!")
        return
    
    print("\n⏰ Starting continuous monitoring...")
    print("   Press Ctrl+C to stop\n")
    
    iteration = 0
    
    try:
        while True:
            iteration += 1
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            print(f"\n{'='*70}")
            print(f"🔄 Check #{iteration} - {timestamp}")
            print(f"{'='*70}")
            
            # Process queue
            process_queue_with_git()
            
            # Wait before next check
            print(f"\n⏳ Waiting {check_interval}s for next check...")
            print(f"   (Next check at {datetime.now().strftime('%H:%M:%S')})")
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Monitoring stopped by user")
        print("="*70)
        print("📊 Final Statistics:")
        
        # Count results
        results = len([f for f in os.listdir(RESULT_DIR) if f.endswith('.json')])
        backups = len(os.listdir(LOCAL_BACKUP_DIR)) if os.path.exists(LOCAL_BACKUP_DIR) else 0
        
        print(f"   Total results: {results}")
        print(f"   Local backups: {backups}")
        print("="*70 + "\n")

# ==================== SINGLE RUN MODE ====================

def run_once():
    """
    Process queue once and exit
    (Good for scheduled/cron jobs)
    """
    
    print("\n" + "="*70)
    print("🎬 GIT QUEUE PROCESSOR - SINGLE RUN MODE")
    print("="*70 + "\n")
    
    # Initialize Git
    if not git_init_clone():
        print("❌ Git initialization failed!")
        return False
    
    # Process queue
    success = process_queue_with_git()
    
    print("\n✅ Single run completed!")
    return success

# ==================== MAIN ENTRY POINT ====================

if __name__ == "__main__":
    import sys
    
    # Check for command line arguments
    mode = sys.argv[1] if len(sys.argv) > 1 else "continuous"
    
    if mode == "once":
        # Single run mode
        run_once()
    elif mode == "continuous":
        # Continuous monitoring mode
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else POLL_INTERVAL
        run_continuous_monitoring(check_interval=interval)
    else:
        print("Usage:")
        print("  python git_queue_processor.py once          # Run once and exit")
        print("  python git_queue_processor.py continuous 30  # Monitor every 30 seconds")
        print("  python git_queue_processor.py               # Monitor every 10 seconds (default)")
