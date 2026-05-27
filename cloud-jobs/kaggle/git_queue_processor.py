"""
🔄 Git Queue Processor v2.0
"""
import os, json, time, shutil, subprocess, sys
from datetime import datetime

GITHUB_TOKEN = ""; GIT_USER_EMAIL = ""; GIT_USER_NAME = ""
GIT_REPO_URL = ""; GIT_BRANCH = "main"; NEXT_GITHUB_REPO = ""

def setup_secrets():
    global GITHUB_TOKEN, GIT_USER_EMAIL, GIT_USER_NAME, GIT_REPO_URL, GIT_BRANCH, NEXT_GITHUB_REPO
    print("🔐 Loading Kaggle Secrets...")
    try:
        from kaggle_secrets import UserSecretsClient
        s = UserSecretsClient()
        GITHUB_TOKEN   = s.get_secret("GITHUB_TOKEN")
        GIT_USER_EMAIL = s.get_secret("GIT_MAIL")
        GIT_USER_NAME  = s.get_secret("GIT_NAME")
        try: NEXT_GITHUB_REPO = s.get_secret("NEXT_GITHUB_REPO")
        except: NEXT_GITHUB_REPO = "KhambhanSingh/khama-dev-ai-cloud-jobs"
        GIT_BRANCH   = "main"
        GIT_REPO_URL = f"https://github.com/{NEXT_GITHUB_REPO}.git"
        print(f"✅ Secrets loaded!\n   Repo: {GIT_REPO_URL}\n   User: {GIT_USER_NAME} <{GIT_USER_EMAIL}>")
        return True
    except Exception as e:
        print(f"❌ Secrets failed: {e}"); return False
        
BASE_DIR   = "cloud-jobs"; QUEUE_DIR  = f"{BASE_DIR}/queue"
VIDEO_DIR  = f"{BASE_DIR}/video"; RESULT_DIR = f"{BASE_DIR}/result"
BACKUP_DIR = f"{BASE_DIR}/local_backup"
for _d in [QUEUE_DIR, VIDEO_DIR, RESULT_DIR, BACKUP_DIR]:
    os.makedirs(_d, exist_ok=True)

def _run(cmd, check=True):
    return subprocess.run(cmd, capture_output=True, text=True, check=check)

def git_configure():
    print("🔧 Configuring Git...")
    _run(['git', 'config', '--global', 'user.email', GIT_USER_EMAIL])
    _run(['git', 'config', '--global', 'user.name',  GIT_USER_NAME])
    _run(['git', 'config', '--global', 'core.autocrlf', 'false'])
    
    # ✅ Token को सही format में URL में डालें
    auth_url = f"https://x-access-token:{GITHUB_TOKEN}@github.com/{NEXT_GITHUB_REPO}.git"
    
    try: _run(['git', 'remote', 'set-url', 'origin', auth_url])
    except: _run(['git', 'remote', 'add', 'origin', auth_url])
    
    print("✅ Git configured!")

def git_init_or_clone():
    if os.path.exists('.git'):
        print("📁 Git repo already present — configuring...")
        git_configure(); return True
    auth_url = GIT_REPO_URL.replace("https://", f"https://{GITHUB_TOKEN}@")
    print("📁 Directory not empty — git init + pull...")
    try:
        _run(['git','init'])
        _run(["git", "checkout", "-B", GIT_BRANCH])
        _run(['git','remote','add','origin', auth_url])
        git_configure()
        _run(['git','fetch','origin', GIT_BRANCH])
        _run(['git','reset','--hard', f'origin/{GIT_BRANCH}'])
        for _d in [QUEUE_DIR, VIDEO_DIR, RESULT_DIR, BACKUP_DIR]:
            os.makedirs(_d, exist_ok=True)
        print("✅ Repo initialized and pulled!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Git init+pull failed: {e.stderr}"); return False

def git_pull():
    print("\n🔄 Git pull...")
    try:
        _run(['git','fetch','origin', GIT_BRANCH])
        _run(['git','reset','--hard', f'origin/{GIT_BRANCH}'])
        for _d in [QUEUE_DIR, VIDEO_DIR, RESULT_DIR, BACKUP_DIR]:
            os.makedirs(_d, exist_ok=True)
        print("✅ Pull successful!"); return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Pull failed: {e.stderr}"); return False

def git_push(message):
    print(f"\n📤 Git push: {message}")
    try:
        _run(['git', 'add', BASE_DIR])
        _run(['git', 'checkout', '-B', 'main'])
        
        r = subprocess.run(
            ['git', 'commit', '-m', message],
            capture_output=True, text=True
        )
        if r.returncode != 0:
            if "nothing to commit" in (r.stdout + r.stderr):
                print("ℹ️  कुछ भी commit करने को नहीं है")
                return True
            print(f"⚠️  Commit: {r.stderr}")
        
        # ✅ Push में explicitly auth URL use करें
        auth_url = f"https://x-access-token:{GITHUB_TOKEN}@github.com/{NEXT_GITHUB_REPO}.git"
        _run(["git", "push", "-u", auth_url, "main"])
        
        print("✅ Push सफल!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Push फेल: {e.stderr}")
        return False
        
def github_raw_url(rel_path):
    if "/" not in NEXT_GITHUB_REPO: return None
    owner, name = NEXT_GITHUB_REPO.split("/", 1)
    rel = rel_path.lstrip("/").replace("\\", "/")
    return f"https://raw.githubusercontent.com/{owner}/{name}/{GIT_BRANCH}/{rel}"

def write_result(record_id, status, video_url=None, error=None):
    os.makedirs(RESULT_DIR, exist_ok=True)
    payload = {"status": status, "recordId": str(record_id)}
    if video_url: payload["videoUrl"] = video_url
    if error: payload["error"] = str(error)[:4000]
    path = os.path.join(RESULT_DIR, f"job_{record_id}_full.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"📝 Result → {path}"); return path

def get_pending_jobs():
    if not os.path.exists(QUEUE_DIR): return []
    return sorted(f for f in os.listdir(QUEUE_DIR)
                  if f.endswith(".json") and "_full.json" in f)

def clear_job(filename):
    path = os.path.join(QUEUE_DIR, filename)
    try:
        if os.path.exists(path): os.remove(path); print(f"🗑️  Cleared: {filename}")
    except Exception as e: print(f"⚠️  Clear failed: {e}")

def cleanup_work_dir(record_id):
    work = os.path.join("cloud-jobs", "work", str(record_id))
    if os.path.isdir(work):
        try: shutil.rmtree(work); print(f"🧹 Work dir removed")
        except Exception as e: print(f"⚠️  Cleanup: {e}")

def backup_locally(result):
    rid = result["recordId"]; folder = os.path.join(BACKUP_DIR, rid)
    os.makedirs(folder, exist_ok=True)
    try:
        if os.path.exists(result.get("video","")): shutil.copy2(result["video"], os.path.join(folder,f"{rid}.mp4"))
        if os.path.exists(result.get("audio","")): shutil.copy2(result["audio"], os.path.join(folder,f"{rid}.wav"))
        print(f"💾 Backup → {folder}")
    except Exception as e: print(f"⚠️  Backup failed: {e}")

def process_queue_once():
    git_pull()
    # Add working dir to path so video_generator_v2 is found
    if '/kaggle/working' not in sys.path:
        sys.path.insert(0, '/kaggle/working')
    try:
        import video_generator_v2 as vg
        print("✅ video_generator_v2 imported\n")
    except ImportError as e:
        print(f"❌ Cannot import video_generator_v2: {e}"); return

    pending = get_pending_jobs()
    if not pending: print("📭 No jobs in queue"); return

    print(f"\n📦 Found {len(pending)} job(s)")
    print("="*70 + "\n")
    ok = fail = 0; need_push = False

    for job_file in pending:
        job_path = os.path.join(QUEUE_DIR, job_file); job_data = None
        try:
            with open(job_path, "r", encoding="utf-8") as f: job_data = json.load(f)
            if job_data.get("type") != "full_video":
                print(f"⏭️  Skip: {job_file}\n"); continue
            record_id = job_data["recordId"]
            print(f"▶️  Processing: {record_id}\n")
            result = vg.process_job(job_data)
            os.makedirs(VIDEO_DIR, exist_ok=True)
            stable = os.path.join(VIDEO_DIR, f"{record_id}.mp4")
            shutil.copy2(result["video"], stable)
            raw_url = github_raw_url(f"cloud-jobs/video/{record_id}.mp4")
            if not raw_url: raise RuntimeError("NEXT_GITHUB_REPO secret missing!")
            write_result(record_id, "DONE", video_url=raw_url)
            need_push = True
            backup_locally(result); cleanup_work_dir(record_id); clear_job(job_file)
            ok += 1; print(f"✅ Done: {record_id}\n")
        except Exception as e:
            import traceback; print(f"❌ Failed: {job_file}\n{traceback.format_exc()}")
            fail += 1
            rid = (job_data or {}).get("recordId")
            if rid:
                try: write_result(rid, "FAILED", error=str(e)); need_push = True
                except: pass
            failed_dir = os.path.join(BASE_DIR, "failed"); os.makedirs(failed_dir, exist_ok=True)
            try: shutil.move(job_path, os.path.join(failed_dir, job_file))
            except: pass

    print("="*70)
    print(f"📊 ✅ {ok} done   ❌ {fail} failed   📦 {len(pending)} total")
    print("="*70 + "\n")
    if need_push:
        git_push(f"cloud-jobs: {ok} done, {fail} failed — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def run_once():
    print("\n" + "="*70 + "\n🎬 SINGLE RUN\n" + "="*70 + "\n")
    process_queue_once()
    print("✅ Done!")

def run_continuous(interval=30):
    print("\n" + "="*70 + f"\n🚀 CONTINUOUS MODE (every {interval}s)\n" + "="*70 + "\n")
    i = 0
    try:
        while True:
            i += 1; print(f"\n{'='*70}\n🔄 Check #{i} — {datetime.now().strftime('%H:%M:%S')}\n{'='*70}")
            process_queue_once()
            print(f"\n⏳ Next check in {interval}s..."); time.sleep(interval)
    except KeyboardInterrupt: print("\n⏹️  Stopped.")

if __name__ == "__main__":
    if not setup_secrets(): sys.exit(1)
    if not git_init_or_clone(): sys.exit(1)
    clean_args = [a for a in sys.argv[1:] if not a.startswith("/") and not a.endswith(".json")]
    mode = clean_args[0] if clean_args else "once"
    try: interval = int(clean_args[1]) if len(clean_args) > 1 else 30
    except: interval = 30
    run_continuous(interval) if mode == "continuous" else run_once()