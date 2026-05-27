"""
Pinned ML package install/check for Kaggle full_video worker.
Used by git_queue_processor.py and install_kaggle_deps.py.
"""

import importlib.metadata
import os
import subprocess
import sys

PINNED = {
    "huggingface-hub": "0.24.0",
    "transformers": "4.44.2",
    "diffusers": "0.30.3",
    "accelerate": "0.33.0",
}

OPTIONAL_PACKAGES = [
    "edge-tts",
    "pydub",
    "safetensors",
    "Pillow",
]


def marker_path():
    if os.path.isdir("/kaggle/working"):
        return "/kaggle/working/.khama_pinned_deps_ok"
    return os.path.join(os.getcwd(), ".khama_pinned_deps_ok")


def pip_extra_args():
    if sys.version_info >= (3, 11):
        return ["--break-system-packages"]
    return []


def get_installed_version(dist_name):
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def versions_match():
    for name, want in PINNED.items():
        got = get_installed_version(name)
        if got != want:
            return False
    return True


def diagnose_versions():
    parts = []
    for name, want in PINNED.items():
        got = get_installed_version(name) or "not installed"
        tag = "OK" if got == want else "BAD"
        parts.append(f"{name}={got} (want {want}) [{tag}]")
    return ", ".join(parts)


def _verify_subprocess():
    pinned_repr = repr(PINNED)
    script = f"""
import importlib.metadata as m
PINNED = {pinned_repr}
ok = True
for name, want in PINNED.items():
    try:
        got = m.version(name)
    except m.PackageNotFoundError:
        got = None
    if got != want:
        print(f"BAD {{name}}={{got}} want {{want}}")
        ok = False
if ok:
    print("OK")
"""
    r = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )
    out = (r.stdout or "").strip()
    if r.returncode != 0 or not out.startswith("OK"):
        err = (r.stderr or r.stdout or "").strip()
        if err:
            print(f"   verify: {err[-800:]}")
        return False
    return True


def _pip_uninstall_conflicts():
    for name in PINNED:
        subprocess.run(
            [
                sys.executable, "-m", "pip", "uninstall", "-y", name,
                *pip_extra_args(),
            ],
            capture_output=True,
            text=True,
        )


def _pip_install_pinned():
    specs = [f"{name}=={ver}" for name, ver in PINNED.items()]
    cmd = [
        sys.executable, "-m", "pip", "install",
        *specs,
        "--force-reinstall",
        "--no-cache-dir",
        *pip_extra_args(),
    ]
    print("   Running single pip install for all pinned packages...")
    return subprocess.run(cmd, capture_output=True, text=True)


def _pip_install_optional():
    for pkg in OPTIONAL_PACKAGES:
        dist = "pillow" if pkg == "Pillow" else pkg
        try:
            importlib.metadata.version(dist)
            continue
        except importlib.metadata.PackageNotFoundError:
            pass
        print(f"   📦 optional {pkg}...")
        subprocess.run(
            [
                sys.executable, "-m", "pip", "install", "-q", pkg,
                *pip_extra_args(),
            ],
            capture_output=True,
            text=True,
        )


def _touch_marker():
    try:
        with open(marker_path(), "w", encoding="utf-8") as f:
            f.write("ok\n")
    except OSError:
        pass


def ensure_pinned_deps(force=False):
    """
    Install pinned ML stack if needed. Returns True when versions match.
    """
    if not force and versions_match():
        print(f"✅ deps OK (skip install): {diagnose_versions()}")
        _touch_marker()
        _pip_install_optional()
        return True

    print("\n📦 Installing pinned ML packages...")
    _pip_uninstall_conflicts()
    r = _pip_install_pinned()
    if r.returncode != 0:
        print(f"   ❌ pip install failed:\n{(r.stderr or r.stdout or '')[-1200:]}")
        print(f"   Installed: {diagnose_versions()}")
        return False

    if not versions_match() and not _verify_subprocess():
        print(f"\n❌ Pinned packages not active after install.")
        print(f"   {diagnose_versions()}")
        subprocess.run(
            [sys.executable, "-m", "pip", "show", "huggingface-hub"],
            capture_output=False,
        )
        print(
            "\n   Try: python cloud-jobs/kaggle/install_kaggle_deps.py\n"
            "   Then Kernel → Restart & Clear Output → Save Environment (Kaggle Settings)"
        )
        return False

    print(f"✅ Pinned packages active: {diagnose_versions()}")
    _touch_marker()
    _pip_install_optional()
    return True
