"""FFmpeg audio loudness, stereo, optional BGM mix."""

import os
import re
import subprocess


def _run_ffmpeg(cmd):
    subprocess.run(cmd, check=True, capture_output=True)


def finalize_narration_audio(in_wav, out_wav=None):
    """Loudnorm + light compression + stereo 44.1kHz."""
    out_wav = out_wav or in_wav
    tmp = out_wav + ".proc.wav"
    af = (
        "loudnorm=I=-16:TP=-1.5:LRA=11,"
        "acompressor=threshold=-20dB:ratio=4:attack=5:release=50,"
        "aformat=channel_layouts=stereo"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        in_wav,
        "-af",
        af,
        "-ar",
        "44100",
        "-ac",
        "2",
        tmp,
    ]
    _run_ffmpeg(cmd)
    os.replace(tmp, out_wav)
    return out_wav


def _slug_topic(topic):
    return re.sub(r"[^a-z0-9]+", "_", str(topic or "").lower()).strip("_")


def resolve_bgm_path(job_data, repo_root=None):
    """Resolve optional BGM file from topic / style."""
    root = repo_root or os.getcwd()
    bgm_dir = os.path.join(root, "public", "bgm")
    candidates = []
    topic = job_data.get("topic") or ""
    style = job_data.get("videoStyle") or ""
    if topic:
        slug = _slug_topic(topic)
        candidates.append(os.path.join(bgm_dir, f"{slug}.mp3"))
        if "kid" in slug or "story" in slug:
            candidates.append(os.path.join(bgm_dir, "kids_gentle.mp3"))
    if style:
        candidates.append(os.path.join(bgm_dir, f"{_slug_topic(style)}.mp3"))
    candidates.append(os.path.join(bgm_dir, "bgm_kids_gentle.mp3"))
    candidates.append(os.path.join(bgm_dir, "default.mp3"))

    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


def mix_bgm(narration_wav, bgm_path, out_wav=None, bgm_volume=0.15):
    """Mix narration with looped background music."""
    out_wav = out_wav or narration_wav
    if not bgm_path or not os.path.isfile(bgm_path):
        return finalize_narration_audio(narration_wav, out_wav)

    tmp = out_wav + ".mix.wav"
    vol = max(0.05, min(0.25, float(bgm_volume)))
    fc = (
        f"[1:a]aloop=loop=-1:size=2e+09,volume={vol}[bg];"
        f"[0:a][bg]amix=inputs=2:duration=first:dropout_transition=0,"
        f"loudnorm=I=-16:TP=-1.5:LRA=11,aformat=channel_layouts=stereo"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        narration_wav,
        "-i",
        bgm_path,
        "-filter_complex",
        fc,
        "-ar",
        "44100",
        "-ac",
        "2",
        tmp,
    ]
    _run_ffmpeg(cmd)
    os.replace(tmp, out_wav)
    return out_wav
