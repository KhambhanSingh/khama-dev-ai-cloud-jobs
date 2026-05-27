"""
Git Queue Processor — full_video jobs from cloud-jobs/queue.

Kaggle (after Next.js has pushed workers + queue to GitHub):
  !python cloud-jobs/kaggle/git_queue_processor.py once
  !python cloud-jobs/kaggle/git_queue_processor.py continuous 30

Do not run video_generator_v2.py directly; this module imports it.
First kernel run may require Restart & Clear Output after pip installs.
"""

import os, json, time, shutil, subprocess, sys
from datetime import datetime

# ==================== GLOBALS ====================
GITHUB_TOKEN     = ""
GIT_USER_EMAIL   = ""
GIT_USER_NAME    = ""
GIT_REPO_URL     = ""
GIT_BRANCH       = "main"
NEXT_GITHUB_REPO = ""
HF_TOKEN         = ""

# ==================== SECRETS ====================
def setup_secrets():
    global GITHUB_TOKEN, GIT_USER_EMAIL, GIT_USER_NAME
    global GIT_REPO_URL, GIT_BRANCH, NEXT_GITHUB_REPO, HF_TOKEN

    print("🔐 Kaggle Secrets load हो रहे हैं...")
    try:
        from kaggle_secrets import UserSecretsClient
        s = UserSecretsClient()

        GITHUB_TOKEN   = s.get_secret("GITHUB_TOKEN")
        GIT_USER_EMAIL = s.get_secret("GIT_MAIL")
        GIT_USER_NAME  = s.get_secret("GIT_NAME")

        try:
            HF_TOKEN = s.get_secret("HF_TOKEN")
            print("   ✅ HF_TOKEN loaded!")
        except Exception:
            HF_TOKEN = ""
            print("   ⚠️  HF_TOKEN set नहीं है")

        try:
            NEXT_GITHUB_REPO = s.get_secret("NEXT_GITHUB_REPO")
        except Exception:
            NEXT_GITHUB_REPO = "KhambhanSingh/khama-dev-ai-cloud-jobs"

        GIT_BRANCH   = "main"
        GIT_REPO_URL = f"https://github.com/{NEXT_GITHUB_REPO}.git"

        print(f"✅ Secrets loaded!")
        print(f"   Repo : {GIT_REPO_URL}")
        print(f"   User : {GIT_USER_NAME} <{GIT_USER_EMAIL}>")
        return True

    except Exception as e:
        print(f"❌ Secrets load failed: {e}")
        return False

# ==================== DIRECTORIES ====================
BASE_DIR         = "cloud-jobs"
QUEUE_DIR        = f"{BASE_DIR}/queue"
VIDEO_DIR        = f"{BASE_DIR}/video"
RESULT_DIR       = f"{BASE_DIR}/result"
BACKUP_DIR       = f"{BASE_DIR}/local_backup"
WORKER_REPO_DIR  = f"{BASE_DIR}/kaggle"
WORKER_NAMES     = ("git_queue_processor.py", "video_generator_v2.py")

for _d in [QUEUE_DIR, VIDEO_DIR, RESULT_DIR, BACKUP_DIR, WORKER_REPO_DIR]:
    os.makedirs(_d, exist_ok=True)

# ==================== WORKER SYNC ====================
def worker_runtime_dir():
    if os.path.isdir("/kaggle/working"):
        return "/kaggle/working"
    return "."

def strip_notebook_magic(text):
    lines = text.splitlines()
    if lines and lines[0].strip().startswith("%%writefile"):
        return "\n".join(lines[1:]) + ("\n" if text.endswith("\n") else "")
    return text

def install_workers_from_repo():
    """Copy cloud-jobs/kaggle/*.py to runtime dir (strip %%writefile)."""
    runtime = worker_runtime_dir()
    os.makedirs(runtime, exist_ok=True)
    installed = 0

    for name in WORKER_NAMES:
        src = os.path.join(WORKER_REPO_DIR, name)
        if not os.path.isfile(src):
            print(f"⚠️  Worker not in repo: {src}")
            continue
        with open(src, "r", encoding="utf-8") as f:
            body = strip_notebook_magic(f.read())
        dest = os.path.join(runtime, name)
        with open(dest, "w", encoding="utf-8") as f:
            f.write(body)
        installed += 1
        print(f"📥 Installed worker → {dest}")

    if runtime not in sys.path:
        sys.path.insert(0, runtime)
    return installed > 0

# ==================== GIT HELPERS ====================
def _run(cmd, check=True):
    return subprocess.run(cmd, capture_output=True, text=True, check=check)

def get_auth_url():
    return f"https://x-access-token:{GITHUB_TOKEN}@github.com/{NEXT_GITHUB_REPO}.git"

def git_configure():
    print("🔧 Git configure...")
    _run(['git', 'config', '--global', 'user.email', GIT_USER_EMAIL])
    _run(['git', 'config', '--global', 'user.name',  GIT_USER_NAME])
    _run(['git', 'config', '--global', 'core.autocrlf', 'false'])
    _run(['git', 'config', '--global', 'init.defaultBranch', 'main'])
    auth_url = get_auth_url()
    r = _run(['git', 'remote', 'set-url', 'origin', auth_url], check=False)
    if r.returncode != 0:
        _run(['git', 'remote', 'add', 'origin', auth_url])
    print("✅ Git configured!")

def git_init_or_clone():
    if os.path.exists('.git'):
        print("📁 Git repo already present")
        git_configure()
        return True

    print("📁 Non-empty dir — git init + pull...")
    try:
        _run(['git', 'init'])
        # ✅ Branch को explicitly main set करो
        _run(['git', 'checkout', '-b', 'main'], check=False)
        _run(['git', 'symbolic-ref', 'HEAD', 'refs/heads/main'])
        git_configure()
        _run(['git', 'fetch', 'origin', GIT_BRANCH])
        _run(['git', 'reset', '--hard', f'origin/{GIT_BRANCH}'])
        for _d in [QUEUE_DIR, VIDEO_DIR, RESULT_DIR, BACKUP_DIR]:
            os.makedirs(_d, exist_ok=True)
        print("✅ Repo ready!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Git init+pull failed: {e.stderr}")
        return False

def git_pull():
    print("\n🔄 Git pull...")
    try:
        _run(['git', 'fetch', 'origin', GIT_BRANCH])
        _run(['git', 'reset', '--hard', f'origin/{GIT_BRANCH}'])
        for _d in [QUEUE_DIR, VIDEO_DIR, RESULT_DIR, BACKUP_DIR]:
            os.makedirs(_d, exist_ok=True)
        print("✅ Pull successful!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Pull failed: {e.stderr}")
        return False

def git_push(message):
    print(f"\n📤 Git push: {message}")
    try:
        _run(['git', 'add', BASE_DIR])

        r = subprocess.run(
            ['git', 'commit', '-m', message],
            capture_output=True, text=True
        )
        if r.returncode != 0:
            if "nothing to commit" in (r.stdout + r.stderr):
                print("ℹ️  Nothing to commit")
                return True
            print(f"⚠️  Commit: {r.stderr[:100]}")

        # ✅ Explicit auth URL + branch name
        auth_url = get_auth_url()
        push_r = subprocess.run(
            ['git', 'push', auth_url, f'HEAD:refs/heads/{GIT_BRANCH}'],
            capture_output=True, text=True
        )
        if push_r.returncode != 0:
            print(f"❌ Push failed: {push_r.stderr[:300]}")
            return False

        print("✅ Push successful!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Push error: {e.stderr[:200]}")
        return False

# ==================== RESULT WRITER ====================
def github_raw_url(rel_path):
    if "/" not in NEXT_GITHUB_REPO:
        return None
    owner, name = NEXT_GITHUB_REPO.split("/", 1)
    rel = rel_path.lstrip("/").replace("\\", "/")
    return f"https://raw.githubusercontent.com/{owner}/{name}/{GIT_BRANCH}/{rel}"

def write_result(record_id, status, video_url=None, error=None):
    os.makedirs(RESULT_DIR, exist_ok=True)
    payload = {"status": status, "recordId": str(record_id)}
    if video_url: payload["videoUrl"] = video_url
    if error:     payload["error"]    = str(error)[:4000]
    path = os.path.join(RESULT_DIR, f"job_{record_id}_full.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"📝 Result → {path}")
    return path

# ==================== QUEUE ====================
def get_pending_jobs():
    if not os.path.exists(QUEUE_DIR): return []
    return sorted(
        f for f in os.listdir(QUEUE_DIR)
        if f.endswith(".json") and "_full.json" in f
    )

def clear_job(filename):
    path = os.path.join(QUEUE_DIR, filename)
    try:
        if os.path.exists(path): os.remove(path)
        print(f"🗑️  Cleared: {filename}")
    except Exception as e:
        print(f"⚠️  Clear: {e}")

def cleanup_work_dir(record_id):
    work = os.path.join("cloud-jobs", "work", str(record_id))
    if os.path.isdir(work):
        try: shutil.rmtree(work); print("🧹 Work dir removed")
        except Exception as e: print(f"⚠️  Cleanup: {e}")

def backup_locally(result):
    rid    = result["recordId"]
    folder = os.path.join(BACKUP_DIR, rid)
    os.makedirs(folder, exist_ok=True)
    try:
        if os.path.exists(result.get("video", "")):
            shutil.copy2(result["video"], os.path.join(folder, f"{rid}.mp4"))
        if os.path.exists(result.get("audio", "")):
            shutil.copy2(result["audio"], os.path.join(folder, f"{rid}.wav"))
        print(f"💾 Backup → {folder}")
    except Exception as e:
        print(f"⚠️  Backup: {e}")

# ==================== MAIN LOOP ====================
def process_queue_once():
    git_pull()
    install_workers_from_repo()

    runtime = worker_runtime_dir()
    if runtime not in sys.path:
        sys.path.insert(0, runtime)

    try:
        import video_generator_v2 as vg
        print("✅ video_generator_v2 imported\n")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("   video_generator_v2.py को /kaggle/working/ में रखें!")
        return
    except SystemExit as e:
        print(f"🔴 video_generator_v2: {e}")
        print("   Kernel restart करें: Menu → Run → Restart & Clear Output")
        return

    pending = get_pending_jobs()
    if not pending:
        print("📭 Queue empty")
        return

    print(f"\n📦 {len(pending)} job(s) मिली")
    print("="*60 + "\n")

    ok = fail = 0
    need_push = False

    for job_file in pending:
        job_path = os.path.join(QUEUE_DIR, job_file)
        job_data = None

        try:
            with open(job_path, "r", encoding="utf-8") as f:
                job_data = json.load(f)

            if job_data.get("type") != "full_video":
                print(f"⏭️  Skip: {job_file}")
                continue

            record_id = job_data["recordId"]
            print(f"▶️  Processing: {record_id}\n")

            result = vg.process_job(job_data, hf_token=HF_TOKEN)

            os.makedirs(VIDEO_DIR, exist_ok=True)
            stable = os.path.join(VIDEO_DIR, f"{record_id}.mp4")
            shutil.copy2(result["video"], stable)

            raw_url = github_raw_url(f"cloud-jobs/video/{record_id}.mp4")
            if not raw_url:
                raise RuntimeError("NEXT_GITHUB_REPO secret missing!")

            write_result(record_id, "DONE", video_url=raw_url)
            need_push = True
            backup_locally(result)
            cleanup_work_dir(record_id)
            clear_job(job_file)
            ok += 1
            print(f"✅ Done: {record_id}\n")

        except Exception as e:
            import traceback
            print(f"❌ Failed: {job_file}\n{traceback.format_exc()}")
            fail += 1
            rid = (job_data or {}).get("recordId")
            if rid:
                try: write_result(rid, "FAILED", error=str(e)); need_push = True
                except: pass
            failed_dir = os.path.join(BASE_DIR, "failed")
            os.makedirs(failed_dir, exist_ok=True)
            try: shutil.move(job_path, os.path.join(failed_dir, job_file))
            except: pass

    print("="*60)
    print(f"📊 ✅ {ok} done   ❌ {fail} failed   📦 {len(pending)} total")
    print("="*60 + "\n")

    if need_push:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        git_push(f"cloud-jobs: {ok} done, {fail} failed — {ts}")

# ==================== MODES ====================
def run_once():
    print("\n" + "="*60 + "\n🎬 SINGLE RUN\n" + "="*60 + "\n")
    process_queue_once()
    print("✅ Done!")

def run_continuous(interval=30):
    print("\n" + "="*60 + f"\n🚀 CONTINUOUS ({interval}s)\n" + "="*60 + "\n")
    i = 0
    try:
        while True:
            i += 1
            print(f"\n{'='*60}\n🔄 #{i} — {datetime.now().strftime('%H:%M:%S')}\n{'='*60}")
            process_queue_once()
            print(f"\n⏳ {interval}s wait...")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n⏹️  Stopped.")

# ==================== ENTRY POINT ====================
if __name__ == "__main__":
    if not setup_secrets():
        sys.exit(1)
    if not git_init_or_clone():
        sys.exit(1)

    # ✅ Kaggle Jupyter argv fix
    clean_args = [
        a for a in sys.argv[1:]
        if not a.startswith("/") and not a.endswith(".json")
    ]
    mode = clean_args[0] if clean_args else "once"
    try:
        interval = int(clean_args[1]) if len(clean_args) > 1 else 30
    except (ValueError, IndexError):
        interval = 30

    if mode == "continuous":
        run_continuous(interval)
    else:
        run_once()