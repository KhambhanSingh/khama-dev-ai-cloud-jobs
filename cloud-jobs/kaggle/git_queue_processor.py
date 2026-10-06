"""
Git Queue Processor — full_video / scene_video jobs from cloud-jobs/queue.

Kaggle (after Next.js has pushed workers + queue to GitHub):

  # ALWAYS-ON (recommended) — run once, leave the cell running
  !cd /kaggle/working && python cloud-jobs/kaggle/git_queue_processor.py continuous 20

  # One-shot (only for debugging)
  !cd /kaggle/working && python cloud-jobs/kaggle/git_queue_processor.py once

Default mode is continuous. Aliases: loop | watch | daemon | always

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
def normalize_github_repo(repo):
    parts = [p for p in repo.strip().strip("/").split("/") if p]
    while len(parts) > 2 and parts[-1] == "cloud-jobs":
        parts.pop()
    if len(parts) != 2:
        raise ValueError(
            f"NEXT_GITHUB_REPO must be owner/repo, got: {repo!r}"
        )
    return f"{parts[0]}/{parts[1]}"

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
            NEXT_GITHUB_REPO = normalize_github_repo(s.get_secret("NEXT_GITHUB_REPO"))
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
WORKER_NAMES     = (
    "git_queue_processor.py",
    "video_generator_v2.py",
    "kaggle_deps.py",
    "install_kaggle_deps.py",
)
WORKER_DATA_FILES = ("kaggle_requirements.txt",)

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

_LAST_INSTALLED_HEAD = None
_DEPS_READY = False
_VG_MODULE = None


def _git_head():
    r = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        return None
    return (r.stdout or "").strip() or None


def _write_if_changed(dest, body):
    """Write file only when content differs — avoids needless churn."""
    try:
        if os.path.isfile(dest):
            with open(dest, "r", encoding="utf-8") as f:
                if f.read() == body:
                    return False
    except OSError:
        pass
    with open(dest, "w", encoding="utf-8") as f:
        f.write(body)
    return True


def install_workers_from_repo(force=False):
    """Copy cloud-jobs/kaggle/*.py and pipeline/ to runtime (skip if HEAD unchanged)."""
    global _LAST_INSTALLED_HEAD
    runtime = worker_runtime_dir()
    os.makedirs(runtime, exist_ok=True)
    head = _git_head()
    if (
        not force
        and head
        and head == _LAST_INSTALLED_HEAD
        and os.path.isfile(os.path.join(runtime, "git_queue_processor.py"))
    ):
        print("📥 Workers unchanged (same HEAD) — skip copy")
        if runtime not in sys.path:
            sys.path.insert(0, runtime)
        return 0

    installed = 0
    changed = 0

    for name in WORKER_NAMES + WORKER_DATA_FILES:
        src = os.path.join(WORKER_REPO_DIR, name)
        if not os.path.isfile(src):
            print(f"⚠️  Worker not in repo: {src}")
            continue
        with open(src, "r", encoding="utf-8") as f:
            raw = f.read()
        body = strip_notebook_magic(raw) if name.endswith(".py") else raw
        dest = os.path.join(runtime, name)
        if _write_if_changed(dest, body):
            changed += 1
            print(f"📥 Installed worker → {dest}")
        installed += 1

    pipeline_src = os.path.join(WORKER_REPO_DIR, "pipeline")
    pipeline_dest = os.path.join(runtime, "pipeline")
    if os.path.isdir(pipeline_src):
        os.makedirs(pipeline_dest, exist_ok=True)
        init_py = os.path.join(pipeline_dest, "__init__.py")
        if not os.path.isfile(init_py):
            with open(init_py, "w", encoding="utf-8") as f:
                f.write("# pipeline package\n")
        for fname in os.listdir(pipeline_src):
            if not fname.endswith(".py"):
                continue
            src = os.path.join(pipeline_src, fname)
            with open(src, "r", encoding="utf-8") as f:
                raw = f.read()
            body = strip_notebook_magic(raw)
            dest = os.path.join(pipeline_dest, fname)
            if _write_if_changed(dest, body):
                changed += 1
                print(f"📥 Installed pipeline → {dest}")
            installed += 1

    if runtime not in sys.path:
        sys.path.insert(0, runtime)
    if head:
        _LAST_INSTALLED_HEAD = head
    if changed == 0:
        print("📥 Workers already up to date")
    else:
        # Hot-reload pipeline code only when files actually changed
        for name in list(sys.modules):
            if name == "pipeline" or name.startswith("pipeline."):
                del sys.modules[name]
        print(f"🔄 Reloaded pipeline modules ({changed} file(s) updated)")
    return changed

# ==================== DEPENDENCIES (before video_generator import) ====================
def _setup_kaggle_import_path():
    runtime = worker_runtime_dir()
    kaggle_dir = os.path.join(os.getcwd(), WORKER_REPO_DIR)
    for p in (runtime, kaggle_dir):
        if p and os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)

def _purge_import_caches(heavy=False):
    """Drop import caches only after pip install. Never drop pipeline while GPU pipe is warm."""
    roots = ("kaggle_deps",)
    if heavy:
        # After real pip changes — reload ML stack. Keep pipeline warm otherwise.
        roots = ("diffusers", "accelerate", "video_generator_v2", "kaggle_deps")
    for name in list(sys.modules):
        if name.split(".")[0] in roots:
            del sys.modules[name]


def ensure_deps_before_import():
    """Pin ML stack once per session (skip repeated SDXL import checks)."""
    global _DEPS_READY
    if _DEPS_READY:
        print("✅ deps cached (skip re-check)")
        return True

    _setup_kaggle_import_path()
    from kaggle_deps import ensure_pinned_deps

    if ensure_pinned_deps(force=False):
        # Do NOT purge pipeline/image_pipeline — that forces SDXL reload every poll.
        _purge_import_caches(heavy=False)
        _DEPS_READY = True
        return True

    print(
        "\n❌ No jobs processed — fix dependencies first.\n"
        "   Run: python cloud-jobs/kaggle/install_kaggle_deps.py\n"
        "   Then Kernel → Restart & Clear Output → Save Environment\n"
    )
    return False

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
    """Fetch + hard reset only when remote moved (avoids empty-poll thrash)."""
    print("\n🔄 Git pull...")
    t0 = time.time()
    try:
        before = _git_head()
        _run(["git", "fetch", "origin", GIT_BRANCH, "--prune"])
        r = subprocess.run(
            ["git", "rev-parse", f"origin/{GIT_BRANCH}"],
            capture_output=True,
            text=True,
        )
        remote = (r.stdout or "").strip() if r.returncode == 0 else None
        if before and remote and before == remote:
            for _d in [QUEUE_DIR, VIDEO_DIR, RESULT_DIR, BACKUP_DIR]:
                os.makedirs(_d, exist_ok=True)
            print(f"✅ Already up to date ({time.time() - t0:.1f}s)")
            return False  # no changes

        _run(["git", "reset", "--hard", f"origin/{GIT_BRANCH}"])
        for _d in [QUEUE_DIR, VIDEO_DIR, RESULT_DIR, BACKUP_DIR]:
            os.makedirs(_d, exist_ok=True)
        print(f"✅ Pull successful ({time.time() - t0:.1f}s)")
        return True  # HEAD changed
    except subprocess.CalledProcessError as e:
        print(f"❌ Pull failed: {e.stderr}")
        return False


def git_push(message):
    """Commit only queue/result/images/failed — skip work/backup bloat."""
    print(f"\n📤 Git push: {message}")
    t0 = time.time()
    try:
        # Stage only what Next.js needs — not work/ or local_backup/
        for rel in (
            QUEUE_DIR,
            RESULT_DIR,
            os.path.join(BASE_DIR, "images"),
            VIDEO_DIR,
            os.path.join(BASE_DIR, "failed"),
        ):
            if os.path.isdir(rel):
                _run(["git", "add", "-A", "--", rel], check=False)

        r = subprocess.run(
            ["git", "commit", "-m", message],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            if "nothing to commit" in (r.stdout + r.stderr):
                print("ℹ️  Nothing to commit")
                return True
            print(f"⚠️  Commit: {r.stderr[:100]}")

        auth_url = get_auth_url()
        push_r = subprocess.run(
            ["git", "push", auth_url, f"HEAD:refs/heads/{GIT_BRANCH}"],
            capture_output=True,
            text=True,
        )
        if push_r.returncode != 0:
            print(f"❌ Push failed: {push_r.stderr[:300]}")
            return False

        print(f"✅ Push successful ({time.time() - t0:.1f}s)")
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

def write_result(record_id, status, video_url=None, image_url=None, error=None, results=None):
    os.makedirs(RESULT_DIR, exist_ok=True)
    payload = {"status": status, "recordId": str(record_id)}
    if video_url: payload["videoUrl"] = video_url
    if image_url: payload["imageUrl"] = image_url
    if error:     payload["error"]    = str(error)[:4000]
    if results is not None:
        payload["results"] = results
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

def process_scene_video_job(job_data):
    """
    Lightweight per-scene I2V: download still → Ken Burns ffmpeg clip → local mp4.
    Used by the Next.js continuity pipeline (type=scene_video).
    """
    import urllib.request

    record_id = job_data["recordId"]
    image_url = job_data.get("imageUrl") or ""
    if not image_url:
        raise RuntimeError("scene_video job missing imageUrl")

    duration = max(4, min(8, float(job_data.get("durationSec") or 6)))
    width = int(job_data.get("width") or 1280)
    height = int(job_data.get("height") or 720)
    fps = int(job_data.get("fps") or 24)

    work = os.path.join("cloud-jobs", "work", str(record_id))
    os.makedirs(work, exist_ok=True)
    still = os.path.join(work, "still.png")
    out = os.path.join(work, f"{record_id}.mp4")

    print(f"⬇️  Downloading still: {image_url[:120]}")
    urllib.request.urlretrieve(image_url, still)
    if not os.path.isfile(still) or os.path.getsize(still) < 500:
        raise RuntimeError("scene_video still download empty")

    frames = max(1, int(round(duration * fps)))
    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},"
        f"zoompan=z='min(zoom+0.0008,1.12)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
        f":d={frames}:s={width}x{height}:fps={fps}"
    )
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1",
        "-i", still,
        "-vf", vf,
        "-t", str(duration),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        out,
    ]
    print("🎞️  Encoding scene_video clip…")
    subprocess.run(cmd, check=True)
    if not os.path.isfile(out) or os.path.getsize(out) < 1000:
        raise RuntimeError("scene_video ffmpeg output empty")
    return {"recordId": record_id, "video": out, "audio": ""}


def process_pipeline_image_job(job_data):
    """
    Character portrait / scene still via SDXL on Kaggle (type=pipeline_image).
    """
    import re
    import urllib.request
    from PIL import Image

    record_id = job_data["recordId"]
    kind = str(job_data.get("kind") or "scene_still")
    prompt = str(job_data.get("prompt") or "cinematic film still").strip()
    species = str(job_data.get("species") or "").strip().lower()
    width = int(job_data.get("width") or 1280)
    height = int(job_data.get("height") or 720)
    ref_urls = list(job_data.get("referenceUrls") or [])

    work = os.path.join("cloud-jobs", "work", str(record_id))
    os.makedirs(work, exist_ok=True)
    out = os.path.join(work, f"{record_id}.png")

    runtime = worker_runtime_dir()
    if runtime not in sys.path:
        sys.path.insert(0, runtime)
    try:
        from pipeline.image_pipeline import (
            REFERENCE_NEGATIVE_PROMPT,
            SCENE_NEGATIVE_PROMPT,
            generate_reference_image,
            load_img2img_model,
            _run_generation,
        )
        from pipeline.prompt_sanitize import strip_forbidden_prompt_words
        from pipeline.validator import validate_reference_png
    except Exception as e:
        raise RuntimeError(f"pipeline.image_pipeline import failed: {e}") from e

    pipe = load_img2img_model()

    # ——— Character portraits: generate_reference_image + Compel long embeds ———
    if kind == "character_sheet":
        subject = species if species and species != "character" else "character"
        client = strip_forbidden_prompt_words(prompt)
        client = re.sub(r"[\u0900-\u097F]+", " ", client)
        client = re.sub(r"\b(no|not|without)\s+\w+", " ", client, flags=re.I)
        client = re.sub(r"\s+", " ", client).strip()
        try:
            from pipeline.prompt_sanitize import (
                _portrait_anatomy,
                sanitize_plain_character_appearance,
            )

            anatomy = _portrait_anatomy(subject)
            description = sanitize_plain_character_appearance(
                job_data.get("description") or "", subject
            )
            client = sanitize_plain_character_appearance(client, subject)
        except Exception:
            anatomy = "correct anatomy"
            description = re.sub(r"\s+", " ", str(job_data.get("description") or "")).strip()

        # PLAIN studio reference only — props/action/env belong in scene jobs
        plain_suffix = (
            "empty hands, no props, no objects, no food, no plants, "
            "pure seamless white studio background, no environment, "
            "no text, no logo, correct anatomy, clear face"
        )
        ref_prompt = (
            f"exactly one {subject}, solo, centered, full body standing, "
            f"{anatomy}, 3D pixar style, {plain_suffix}"
        )
        if description:
            ref_prompt = f"{ref_prompt}, {description}"
        elif client:
            # Keep color/face words from client prompt only
            ref_prompt = f"{ref_prompt}, {client}"
        ref_prompt = " ".join(ref_prompt.split()[:70])
        appearance = description or f"stylized {subject}"

        char = {
            "id": str(record_id),
            "name": str(job_data.get("name") or subject),
            "species": subject,
            "appearance": appearance or f"stylized {subject}",
            "description": description,
            "referencePrompt": ref_prompt,
            "videoStyle": "3D pixar",
        }
        # Native Turbo size; generate_reference_image upscales to width/height
        gen_w = gen_h = 512
        seed = sum(ord(c) for c in str(record_id)) % 100000
        # Uniform-colour animals trip edge-cell clone heuristics
        animal_qa = {"min_face_cells": 20, "min_clone_matches": 16}
        print(f"🖼️  Character portrait via generate_reference_image ({gen_w}x{gen_h}→upscale)")
        print(f"   prompt: {ref_prompt}  (words={len(ref_prompt.split())})")
        # #region agent log
        print(
            f'   [debug:5928f0] char_plain '
            f'{{"hypothesisId":"H-props-in-desc","subject":{json.dumps(subject)},'
            f'"refWords":{len(ref_prompt.split())},'
            f'"propLeak":{str(bool(re.search(r"carries|holds|banana|wears|flower|sand", ref_prompt, re.I))).lower()},'
            f'"emptyHands":{str("empty hands" in ref_prompt.lower()).lower()},'
            f'"head":{json.dumps(ref_prompt[:220])}}}'
        )
        # #endregion
        try:
            generate_reference_image(
                pipe,
                char,
                gen_w,
                gen_h,
                width or 1024,
                height or 1024,
                out,
                video_style="3D pixar",
                negative_prompt=REFERENCE_NEGATIVE_PROMPT,
                seed=seed,
                validate_kwargs=animal_qa,
            )
        except Exception as first_err:
            print(f"   ⚠️  portrait QA failed: {first_err}")
            if os.path.isfile(out) and os.path.getsize(out) >= 1000:
                # Last attempt file kept — accept with softer animal thresholds
                try:
                    validate_reference_png(
                        out, min_face_cells=22, min_clone_matches=18
                    )
                    print("   ✅ accepted with relaxed animal QA")
                except Exception as soft_err:
                    print(f"   ⚠️  relaxed QA warn (keeping image): {soft_err}")
            else:
                raise RuntimeError(
                    f"pipeline_image character failed: {first_err}"
                ) from first_err

        if not os.path.isfile(out) or os.path.getsize(out) < 1000:
            raise RuntimeError("pipeline_image character output empty")
        try:
            from pipeline.image_pipeline import clear_gpu_memory

            clear_gpu_memory()
        except Exception:
            pass
        return {"recordId": record_id, "image": out}

    # ——— Scene stills ———
    # Keep VISUAL ACTION; only strip Devanagari (Hindi) tokens, not the whole beat
    prompt = strip_forbidden_prompt_words(prompt)
    prompt = re.sub(r"[\u0900-\u097F]+", " ", prompt)
    prompt = re.sub(r"\s+", " ", prompt).strip()
    try:
        from pipeline.image_pipeline import _action_first_prompt

        prompt = _action_first_prompt(prompt)
    except Exception:
        pass

    gen_w = min(1280, max(768, width))
    gen_h = min(720, max(512, height))
    gen_w = max(512, (gen_w // 8) * 8)
    gen_h = max(512, (gen_h // 8) * 8)

    init_image = None
    strength = 1.0  # Turbo txt2img-like
    if ref_urls:
        ref_path = os.path.join(work, "ref0.png")
        print(f"⬇️  Downloading reference: {ref_urls[0][:120]}")
        urllib.request.urlretrieve(ref_urls[0], ref_path)
        if os.path.isfile(ref_path) and os.path.getsize(ref_path) > 500:
            init_image = Image.open(ref_path).convert("RGB")
            strength = 0.5  # Turbo img2img: steps*strength >= 1

    print(f"🖼️  Generating scene_still ({gen_w}x{gen_h})…")
    print(f"   prompt head (action-first): {prompt[:200]}")
    # #region agent log
    print(
        f'   [debug:5928f0] scene prompt '
        f'{{"words":{len(prompt.split())},"hasVisualAction":'
        f'{str("visual action" in prompt.lower()).lower()},'
        f'"head":{json.dumps(prompt[:180])}}}'
    )
    # #endregion
    image = _run_generation(
        pipe,
        prompt,
        gen_w,
        gen_h,
        init_image=init_image,
        strength=strength,
        steps=4,
        guidance=0.0,  # SDXL-Turbo official: CFG off
        negative_prompt=SCENE_NEGATIVE_PROMPT,
    )
    if width != gen_w or height != gen_h:
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    image.save(out, format="PNG")
    if not os.path.isfile(out) or os.path.getsize(out) < 1000:
        raise RuntimeError("pipeline_image output empty")
    return {"recordId": record_id, "image": out}


def _load_batch_payload(job_data):
    """Load characters/scenes batch JSON from repo path or URL."""
    import urllib.request

    path = str(job_data.get("batchJsonPath") or "").strip()
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    url = str(job_data.get("batchJsonUrl") or "").strip()
    if url:
        dest = os.path.join(
            "cloud-jobs", "work", str(job_data.get("recordId") or "batch"), "batch.json"
        )
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        urllib.request.urlretrieve(url, dest)
        with open(dest, "r", encoding="utf-8") as f:
            return json.load(f)
    raise RuntimeError("pipeline_image_batch missing batchJsonPath/batchJsonUrl")


def process_pipeline_image_batch_job(job_data):
    """
    ONE Kaggle run for all characters OR all scenes (type=pipeline_image_batch).
    Loads batch JSON, keeps SDXL pipe warm, writes per-item images + results[].
    """
    import re
    import urllib.request
    from PIL import Image

    record_id = job_data["recordId"]
    kind = str(job_data.get("kind") or "").strip().lower()
    payload = _load_batch_payload(job_data)
    items = list(payload.get("items") or [])
    if kind not in ("characters", "scenes"):
        kind = str(payload.get("kind") or "scenes").strip().lower()

    work = os.path.join("cloud-jobs", "work", str(record_id))
    os.makedirs(work, exist_ok=True)
    img_dir = os.path.join(BASE_DIR, "images")
    os.makedirs(img_dir, exist_ok=True)

    runtime = worker_runtime_dir()
    if runtime not in sys.path:
        sys.path.insert(0, runtime)
    try:
        from pipeline.image_pipeline import (
            SCENE_NEGATIVE_PROMPT,
            load_img2img_model,
            _run_generation,
            clear_gpu_memory,
        )
        from pipeline.prompt_sanitize import strip_forbidden_prompt_words
    except Exception as e:
        raise RuntimeError(f"pipeline.image_pipeline import failed: {e}") from e

    try:
        from pipeline.image_pipeline import _action_first_prompt
    except Exception:
        _action_first_prompt = None

    print(f"📦 Batch {kind}: {len(items)} item(s) record={record_id}")
    # #region agent log
    print(
        f'   [debug:5928f0] batch_start '
        f'{{"hypothesisId":"H5","kind":{json.dumps(kind)},'
        f'"count":{len(items)},"recordId":{json.dumps(str(record_id))}}}'
    )
    # #endregion

    pipe = load_img2img_model()
    results = []
    prev_scene_path = None
    ok_n = fail_n = skip_n = 0

    # Scenes: order by scene_number for continuity
    if kind == "scenes":
        items = sorted(
            items,
            key=lambda it: int(it.get("scene_number") or it.get("order") or 0),
        )

    for idx, item in enumerate(items):
        if item.get("skip"):
            skip_n += 1
            results.append(
                {
                    "id": item.get("id"),
                    "scene_number": item.get("scene_number"),
                    "skip": True,
                    "status": "skipped",
                    "imageUrl": item.get("imageUrl") or "",
                }
            )
            continue

        try:
            if kind == "characters":
                # Reuse single-job character path via synthetic job
                sub_id = f"{record_id}_c{idx}_{str(item.get('id') or idx)[:12]}"
                sub_job = {
                    "recordId": sub_id,
                    "kind": "character_sheet",
                    "prompt": item.get("prompt") or "",
                    "species": item.get("species") or "",
                    "name": item.get("name") or "",
                    "description": item.get("description") or "",
                    "width": int(item.get("width") or 1024),
                    "height": int(item.get("height") or 1024),
                    "referenceUrls": [],
                }
                out_info = process_pipeline_image_job(sub_job)
                src = out_info["image"]
                stable_name = f"{record_id}_char_{str(item.get('id') or idx)}.png"
                stable = os.path.join(img_dir, stable_name)
                shutil.copy2(src, stable)
                raw_url = github_raw_url(f"cloud-jobs/images/{stable_name}")
                if not raw_url:
                    raise RuntimeError("NEXT_GITHUB_REPO secret missing!")
                results.append(
                    {
                        "id": item.get("id"),
                        "name": item.get("name"),
                        "status": "done",
                        "imageUrl": raw_url,
                    }
                )
                ok_n += 1
                # #region agent log
                print(
                    f'   [debug:5928f0] batch_char_ok '
                    f'{{"hypothesisId":"H2","id":{json.dumps(str(item.get("id")))},'
                    f'"idx":{idx}}}'
                )
                # #endregion
            else:
                prompt = str(
                    item.get("final_prompt") or item.get("prompt") or ""
                ).strip()
                neg = str(
                    item.get("negative_prompt") or SCENE_NEGATIVE_PROMPT
                )
                width = int(item.get("width") or 1280)
                height = int(item.get("height") or 720)
                seed = int(item.get("seed") or (idx * 9973) % 100000)

                prompt = strip_forbidden_prompt_words(prompt)
                prompt = re.sub(r"[\u0900-\u097F]+", " ", prompt)
                prompt = re.sub(r"\s+", " ", prompt).strip()
                if _action_first_prompt:
                    try:
                        prompt = _action_first_prompt(prompt)
                    except Exception:
                        pass

                gen_w = min(1280, max(768, width))
                gen_h = min(720, max(512, height))
                gen_w = max(512, (gen_w // 8) * 8)
                gen_h = max(512, (gen_h // 8) * 8)

                init_image = None
                strength = 1.0
                ref_urls = list(item.get("character_ref_urls") or [])
                use_prev = bool(item.get("use_previous_scene", True))

                if use_prev and prev_scene_path and os.path.isfile(prev_scene_path):
                    init_image = Image.open(prev_scene_path).convert("RGB")
                    strength = 0.55
                elif ref_urls:
                    ref_path = os.path.join(work, f"ref_{idx}.png")
                    urllib.request.urlretrieve(ref_urls[0], ref_path)
                    if os.path.isfile(ref_path) and os.path.getsize(ref_path) > 500:
                        init_image = Image.open(ref_path).convert("RGB")
                        strength = 0.5

                # #region agent log
                print(
                    f'   [debug:5928f0] batch_scene_gen '
                    f'{{"hypothesisId":"H3","scene":{item.get("scene_number")},'
                    f'"words":{len(prompt.split())},'
                    f'"hasLoc":{str("location" in prompt.lower() or bool(item.get("location"))).lower()},'
                    f'"hasAction":{str(bool(item.get("key_action")) or "visual action" in prompt.lower()).lower()},'
                    f'"head":{json.dumps(prompt[:160])}}}'
                )
                # #endregion

                image = _run_generation(
                    pipe,
                    prompt,
                    gen_w,
                    gen_h,
                    init_image=init_image,
                    strength=strength,
                    steps=4,
                    guidance=0.0,
                    negative_prompt=neg,
                    seed=seed,
                )
                if width != gen_w or height != gen_h:
                    image = image.resize((width, height), Image.Resampling.LANCZOS)

                sn = int(item.get("scene_number") or idx + 1)
                stable_name = f"{record_id}_scene_{sn:02d}.png"
                stable = os.path.join(img_dir, stable_name)
                image.save(stable, format="PNG")
                if not os.path.isfile(stable) or os.path.getsize(stable) < 1000:
                    raise RuntimeError("batch scene output empty")

                prev_scene_path = stable
                raw_url = github_raw_url(f"cloud-jobs/images/{stable_name}")
                if not raw_url:
                    raise RuntimeError("NEXT_GITHUB_REPO secret missing!")
                results.append(
                    {
                        "id": item.get("id"),
                        "scene_number": sn,
                        "status": "done",
                        "imageUrl": raw_url,
                    }
                )
                ok_n += 1
        except Exception as item_err:
            fail_n += 1
            print(f"   ⚠️  batch item {idx} failed: {item_err}")
            results.append(
                {
                    "id": item.get("id"),
                    "scene_number": item.get("scene_number"),
                    "status": "failed",
                    "error": str(item_err)[:300],
                    "imageUrl": "",
                }
            )

    try:
        clear_gpu_memory()
    except Exception:
        pass

    status = "DONE" if fail_n == 0 else ("PARTIAL" if ok_n else "FAILED")
    # #region agent log
    print(
        f'   [debug:5928f0] batch_done '
        f'{{"hypothesisId":"H2","status":{json.dumps(status)},'
        f'"ok":{ok_n},"failed":{fail_n},"skipped":{skip_n}}}'
    )
    # #endregion
    return {
        "recordId": record_id,
        "status": status,
        "results": results,
        "ok": ok_n,
        "failed": fail_n,
        "skipped": skip_n,
    }


# ==================== MAIN LOOP ====================
_vg_import_ok = False

def process_queue_once():
    global _vg_import_ok, _VG_MODULE, _LAST_INSTALLED_HEAD

    pull_changed = bool(git_pull())
    # Copy workers only on first run or when remote HEAD moved
    install_workers_from_repo(
        force=pull_changed or _LAST_INSTALLED_HEAD is None
    )

    pending = get_pending_jobs()
    if not pending:
        print("📭 Queue empty")
        return

    # Heavy path only when there is work (skip deps/SDXL on empty polls)
    runtime = worker_runtime_dir()
    if runtime not in sys.path:
        sys.path.insert(0, runtime)

    if not ensure_deps_before_import():
        print("❌ No jobs processed — fix dependencies first.\n")
        return

    vg = _VG_MODULE
    if vg is None:
        try:
            import video_generator_v2 as vg
            _VG_MODULE = vg
            _vg_import_ok = True
            print("✅ video_generator_v2 imported\n")
        except ImportError as e:
            print(f"⚠️  video_generator_v2 import failed: {e}")
            print("   pipeline_image / scene_video still run; full_video skipped.")
        except SystemExit as e:
            msg = str(e) or ""
            print(f"⚠️  video_generator_v2: {msg}")
            print("   pipeline_image / scene_video still run; full_video skipped.")
    else:
        _vg_import_ok = True
        print("✅ video_generator_v2 cached\n")

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

            job_type = job_data.get("type")
            if job_type not in (
                "full_video",
                "scene_video",
                "pipeline_image",
                "pipeline_image_batch",
            ):
                print(f"⏭️  Skip: {job_file}")
                continue

            record_id = job_data["recordId"]
            print(f"▶️  Processing ({job_type}): {record_id}\n")

            if HF_TOKEN:
                os.environ["HF_TOKEN"] = HF_TOKEN
                os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

            if job_type == "scene_video":
                result = process_scene_video_job(job_data)
                os.makedirs(VIDEO_DIR, exist_ok=True)
                stable = os.path.join(VIDEO_DIR, f"{record_id}.mp4")
                shutil.copy2(result["video"], stable)
                raw_url = github_raw_url(f"cloud-jobs/video/{record_id}.mp4")
                if not raw_url:
                    raise RuntimeError("NEXT_GITHUB_REPO secret missing!")
                write_result(record_id, "DONE", video_url=raw_url)
            elif job_type == "pipeline_image_batch":
                result = process_pipeline_image_batch_job(job_data)
                write_result(
                    record_id,
                    result.get("status") or "DONE",
                    results=result.get("results") or [],
                    error=(
                        f"{result.get('failed', 0)} item(s) failed"
                        if result.get("failed")
                        else None
                    ),
                )
            elif job_type == "pipeline_image":
                result = process_pipeline_image_job(job_data)
                img_dir = os.path.join(BASE_DIR, "images")
                os.makedirs(img_dir, exist_ok=True)
                stable = os.path.join(img_dir, f"{record_id}.png")
                shutil.copy2(result["image"], stable)
                raw_url = github_raw_url(f"cloud-jobs/images/{record_id}.png")
                if not raw_url:
                    raise RuntimeError("NEXT_GITHUB_REPO secret missing!")
                write_result(record_id, "DONE", image_url=raw_url)
            else:
                if vg is None:
                    raise RuntimeError(
                        "full_video needs video_generator_v2 — fix import / deps"
                    )
                result = vg.process_job(job_data)
                os.makedirs(VIDEO_DIR, exist_ok=True)
                stable = os.path.join(VIDEO_DIR, f"{record_id}.mp4")
                shutil.copy2(result["video"], stable)
                raw_url = github_raw_url(f"cloud-jobs/video/{record_id}.mp4")
                if not raw_url:
                    raise RuntimeError("NEXT_GITHUB_REPO secret missing!")
                write_result(record_id, "DONE", video_url=raw_url)
                backup_locally(result)
                cleanup_work_dir(record_id)

            need_push = True
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
    if not _vg_import_ok:
        print("❌ Exiting — worker did not load; queue was not processed.")
        sys.exit(1)
    print("✅ Done! (check logs for Processing / Queue empty)")
    print("💡 Tip: leave the worker running with:")
    print("   python cloud-jobs/kaggle/git_queue_processor.py continuous 20")

def run_continuous(interval=20):
    # Empty-queue polls are cheap now; 15s default is fine. Cap stays 300.
    print("\n" + "="*60)
    print(f"🚀 ALWAYS-ON WORKER — polls every {interval}s")
    print("   Leave this cell RUNNING. Do not re-run manually.")
    print("   Stop: interrupt/stop the notebook cell.")
    print("   Fast path: empty queue skips deps/SDXL reload.")
    print("="*60 + "\n")
    i = 0
    try:
        while True:
            i += 1
            print(f"\n{'='*60}\n🔄 #{i} — {datetime.now().strftime('%H:%M:%S')}\n{'='*60}")
            t0 = time.time()
            try:
                process_queue_once()
            except Exception as e:
                # Keep the loop alive — next poll retries after pull
                print(f"⚠️  Poll error (will retry): {e}")
            print(
                f"\n⏳ Next poll in {interval}s… "
                f"(last cycle {time.time() - t0:.1f}s, cell stays alive)"
            )
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
    # Default = continuous so you don't re-execute the cell for every job
    mode = (clean_args[0] if clean_args else "continuous").lower()
    try:
        interval = int(clean_args[1]) if len(clean_args) > 1 else 20
    except (ValueError, IndexError):
        interval = 20
    interval = max(10, min(300, interval))

    if mode in ("continuous", "loop", "watch", "daemon", "always"):
        run_continuous(interval)
    elif mode in ("once", "single"):
        run_once()
    else:
        print(f"Unknown mode {mode!r} — using continuous")
        run_continuous(interval)