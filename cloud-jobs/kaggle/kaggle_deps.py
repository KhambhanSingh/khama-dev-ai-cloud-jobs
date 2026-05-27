"""
Pinned ML package install/check for Kaggle full_video worker.
Aligned with video_generator_v2.py (diffusers + torch/torchvision only).
Used by git_queue_processor.py and install_kaggle_deps.py.
"""

import importlib.metadata
import os
import re
import subprocess
import sys

try:
    from packaging.version import Version
except ImportError:
    Version = None  # type: ignore[misc, assignment]

DEPS_POLICY = "v3"

PINNED = {
    "diffusers": "0.30.3",
}

OPTIONAL_PACKAGES = [
    "gtts",
    "pydub",
    "moviepy",
    "Pillow",
    "safetensors",
]

OPTIONAL_PIP_NAMES = {
    "gtts": "gTTS",
    "Pillow": "Pillow",
}


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
    print("   Running pip install for pinned packages...")
    return subprocess.run(cmd, capture_output=True, text=True)


def _optional_import_ok(pkg):
    mod = "PIL" if pkg == "Pillow" else pkg
    r = _run_python_script(f"import {mod}\nprint('OK')")
    return r.returncode == 0 and (r.stdout or "").strip().startswith("OK")


def _pip_install_optional():
    for pkg in OPTIONAL_PACKAGES:
        pip_name = OPTIONAL_PIP_NAMES.get(pkg, pkg)
        if _optional_import_ok(pkg):
            continue
        print(f"   optional {pip_name}...")
        subprocess.run(
            [
                sys.executable, "-m", "pip", "install", "-q", pip_name,
                *pip_extra_args(),
            ],
            capture_output=True,
            text=True,
        )


def _touch_marker():
    try:
        with open(marker_path(), "w", encoding="utf-8") as f:
            f.write(f"{DEPS_POLICY}\n")
    except OSError:
        pass


def _run_python_script(script):
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )


def _torch_version_subprocess():
    r = _run_python_script("import torch; print(torch.__version__)")
    if r.returncode != 0:
        return None
    return (r.stdout or "").strip().split()[0] if r.stdout else None


def _torch_base_version(torch_ver):
    if not torch_ver:
        return None
    return torch_ver.split("+")[0].strip()


def _torchvision_spec_for_torch(torch_ver):
    """Pick torchvision wheel compatible with Kaggle preinstalled torch (do not reinstall torch)."""
    base = _torch_base_version(torch_ver)
    if not base:
        return "torchvision"
    if Version is None:
        return "torchvision"
    try:
        v = Version(base)
    except Exception:
        return "torchvision"
    if v >= Version("2.12.0"):
        spec = "torchvision==0.27.0"
    elif v >= Version("2.5.0"):
        spec = "torchvision==0.20.1"
    elif v >= Version("2.4.0"):
        spec = "torchvision==0.19.1"
    elif v >= Version("2.3.0"):
        spec = "torchvision==0.18.1"
    elif v >= Version("2.2.0"):
        spec = "torchvision==0.17.2"
    elif v >= Version("2.1.0"):
        spec = "torchvision==0.16.2"
    else:
        spec = "torchvision"
    print(f"   torchvision map: torch {base} (parsed {v}) -> {spec}")
    return spec


def _pytorch_index_url_for_torch(torch_ver):
    m = re.search(r"\+(cu\d+)", torch_ver or "")
    cuda = m.group(1) if m else "cu121"
    return f"https://download.pytorch.org/whl/{cuda}"


def _reassert_pinned_diffusers():
    if versions_match():
        return True
    print("   Re-installing diffusers after torchvision fix...")
    r = _pip_install_pinned()
    if r.returncode != 0:
        print(f"   reassert pip failed:\n{(r.stderr or r.stdout or '')[-800:]}")
        return False
    return versions_match() or _verify_subprocess()


def _verify_torchvision_nms_subprocess():
    script = """
import torch
import torchvision
from torchvision.ops import nms
print("OK")
"""
    r = _run_python_script(script)
    out = (r.stdout or "").strip()
    if r.returncode == 0 and out.startswith("OK"):
        return True
    err = (r.stderr or r.stdout or "").strip()
    if err:
        print(f"   torchvision verify: {err[-800:]}")
    return False


def _pip_install_torchvision_compat():
    torch_ver = _torch_version_subprocess()
    if not torch_ver:
        print("   torch not available (GPU image required)")
        return False
    spec = _torchvision_spec_for_torch(torch_ver)
    index_url = _pytorch_index_url_for_torch(torch_ver)
    print(f"   Installing {spec} for torch {torch_ver} (index {index_url}, --no-deps)...")
    r = subprocess.run(
        [
            sys.executable, "-m", "pip", "install", spec,
            "--index-url", index_url,
            "--force-reinstall", "--no-deps", "--no-cache-dir",
            *pip_extra_args(),
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        print(f"   torchvision install failed:\n{(r.stderr or r.stdout or '')[-800:]}")
        return False
    if not _verify_torchvision_nms_subprocess():
        return False
    return _reassert_pinned_diffusers()


def _ensure_torchvision():
    if _verify_torchvision_nms_subprocess():
        return True
    print("\n📦 Fixing torchvision (torchvision::nms) for diffusers...")
    return _pip_install_torchvision_compat()


def _verify_diffusers_import_subprocess():
    script = """
from diffusers import StableDiffusionXLPipeline
print("OK")
"""
    r = _run_python_script(script)
    out = (r.stdout or "").strip()
    if r.returncode == 0 and out.startswith("OK"):
        print("   diffusers SDXL import OK")
        return True
    err = (r.stderr or r.stdout or "").strip()
    if err:
        print(f"   diffusers import failed: {err[-1200:]}")
    return False


def _ensure_runtime_imports():
    if not _ensure_torchvision():
        print("\n❌ torchvision/torch mismatch — cannot load diffusers.")
        return False
    if not _verify_diffusers_import_subprocess():
        print("\n❌ diffusers import check failed — fix deps before processing queue.")
        return False
    return True


def ensure_pinned_deps(force=False):
    """
    Install pinned ML stack if needed. Returns True when versions match and imports OK.
    """
    print(
        f"Kaggle deps policy {DEPS_POLICY} "
        "(diffusers-only; no huggingface-hub downgrade)"
    )

    if not force and versions_match():
        print(f"✅ deps OK (skip install): {diagnose_versions()}")
        _touch_marker()
        _pip_install_optional()
        return _ensure_runtime_imports()

    print("\n📦 Installing pinned ML packages...")
    _pip_uninstall_conflicts()
    r = _pip_install_pinned()
    if r.returncode != 0:
        print(f"   ❌ pip install failed:\n{(r.stderr or r.stdout or '')[-1200:]}")
        print(f"   Installed: {diagnose_versions()}")
        return False

    if not versions_match() and not _verify_subprocess():
        print("\n❌ Pinned packages not active after install.")
        print(f"   {diagnose_versions()}")
        print(
            "\n   Try: python cloud-jobs/kaggle/install_kaggle_deps.py\n"
            "   Then Kernel → Restart & Clear Output → Save Environment (Kaggle Settings)"
        )
        return False

    print(f"✅ Pinned packages active: {diagnose_versions()}")
    _touch_marker()
    _pip_install_optional()
    return _ensure_runtime_imports()
