"""FFmpeg audio loudness, stereo, optional BGM mix."""

import os
import re
import subprocess


def _run_ffmpeg(cmd):
    subprocess.run(cmd, check=True, capture_output=True)


def validate_voice_present(audio_path, min_max_volume_db=-35.0):
    """
    Run ffmpeg volumedetect on the mixed audio and raise if it appears silent.
    BGM-only or silent audio will have max_volume well below -35 dBFS after
    loudnorm; a mix containing real voice narration will peak much higher.
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"audio file missing for voice check: {audio_path}")
    cmd = [
        "ffmpeg",
        "-i",
        audio_path,
        "-af",
        "volumedetect",
        "-vn",
        "-sn",
        "-dn",
        "-f",
        "null",
        "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    m = re.search(r"max_volume:\s*([-\d.]+)\s*dB", result.stderr)
    if m:
        max_vol = float(m.group(1))
        if max_vol < min_max_volume_db:
            raise ValueError(
                f"Audio appears silent or BGM-only: max_volume={max_vol:.1f} dB "
                f"(threshold {min_max_volume_db} dB) — TTS may have failed"
            )


def finalize_narration_audio(in_wav, out_wav=None):
    """Loudnorm + light compression + stereo 44.1kHz. Target -14 LUFS (YouTube standard)."""
    out_wav = out_wav or in_wav
    tmp = out_wav + ".proc.wav"
    af = (
        "loudnorm=I=-14:TP=-1.5:LRA=11,"
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
    """Mix narration (voice at 100%) with looped background music at bgm_volume.
    Voice is mapped as input 0 so amix duration=first preserves narration length.
    Target loudness -14 LUFS (YouTube standard).
    """
    out_wav = out_wav or narration_wav
    if not bgm_path or not os.path.isfile(bgm_path):
        return finalize_narration_audio(narration_wav, out_wav)

    tmp = out_wav + ".mix.wav"
    vol = max(0.05, min(0.25, float(bgm_volume)))
    fc = (
        f"[1:a]aloop=loop=-1:size=2e+09,volume={vol}[bg];"
        f"[0:a]volume=1.0[voice];"
        f"[voice][bg]amix=inputs=2:duration=first:dropout_transition=0,"
        f"loudnorm=I=-14:TP=-1.5:LRA=11,aformat=channel_layouts=stereo"
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
