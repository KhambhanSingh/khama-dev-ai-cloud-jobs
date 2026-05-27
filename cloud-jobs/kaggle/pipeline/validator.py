"""Validation and retry helpers."""

import json
import os
import subprocess
import time

from .logging_util import log_stage

MIN_SCENE_PNG_BYTES = 10_000
MIN_SCENE_STD_DEV = 12.0
GREY_RGB = (128, 128, 128)


def retry_stage(fn, stage, record_id, max_attempts=3, delay_sec=2.0):
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as e:
            last_err = e
            log_stage(
                stage,
                record_id,
                message=f"attempt {attempt}/{max_attempts} failed: {e}",
                level="ERROR",
            )
            if attempt < max_attempts:
                time.sleep(delay_sec * attempt)
    raise last_err


def validate_beats(beats):
    if not beats or not isinstance(beats, list):
        raise ValueError("beats missing or empty")
    for i, b in enumerate(beats):
        if not str(b.get("narrationText", "")).strip() and not str(
            b.get("visualPrompt", "")
        ).strip():
            raise ValueError(f"beat {i} has no narration or visual prompt")


def validate_audio(path, min_duration=0.5):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"audio missing: {path}")
    if os.path.getsize(path) < 1000:
        raise ValueError(f"audio too small: {path}")


def validate_scene_png(path, min_bytes=MIN_SCENE_PNG_BYTES, min_std=MIN_SCENE_STD_DEV):
    """Reject missing, tiny, or flat grey placeholder images."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"scene image missing: {path}")
    size = os.path.getsize(path)
    if size < min_bytes:
        raise ValueError(f"scene image too small ({size} bytes): {path}")

    try:
        from PIL import Image, ImageStat

        img = Image.open(path).convert("RGB")
        stat = ImageStat.Stat(img)
        stddev = sum(stat.stddev) / 3.0
        if stddev < min_std:
            raise ValueError(
                f"scene image appears flat/grey (stddev={stddev:.1f}): {path}"
            )
        mean = tuple(int(x) for x in stat.mean)
        if all(abs(m - GREY_RGB[i]) < 8 for i, m in enumerate(mean)) and stddev < 20:
            raise ValueError(f"scene image is uniform grey canvas: {path}")
    except ImportError:
        pass
    return path


def validate_scene_images(paths, min_count=1, audio_duration_sec=None):
    if not paths:
        raise ValueError("no scene images generated")
    for p in paths:
        validate_scene_png(p)
    if audio_duration_sec and audio_duration_sec > 0:
        expected = max(min_count, int(audio_duration_sec / 5) + 1)
        if len(paths) < expected:
            raise ValueError(
                f"too few scene images ({len(paths)}) for {audio_duration_sec:.1f}s audio "
                f"(expected >= {expected})"
            )


def validate_scene_clips(paths):
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f"scene clips missing: {missing[:3]}")


def validate_video(path, min_bytes=5000):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"video missing: {path}")
    if os.path.getsize(path) < min_bytes:
        raise ValueError(f"video too small: {path}")


def validate_video_output(path, min_bitrate_kbps=500, min_duration=1.0):
    """ffprobe validation: duration, bitrate, CFR frame rate."""
    validate_video(path, min_bytes=50_000)

    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        path,
    ]
    try:
        raw = subprocess.run(cmd, check=True, capture_output=True, text=True)
        data = json.loads(raw.stdout or "{}")
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        raise ValueError(f"ffprobe failed for {path}: {e}") from e

    fmt = data.get("format") or {}
    duration = float(fmt.get("duration") or 0)
    if duration < min_duration:
        raise ValueError(f"video duration too short ({duration:.2f}s): {path}")

    bitrate = int(fmt.get("bit_rate") or 0) / 1000
    if bitrate > 0 and bitrate < min_bitrate_kbps:
        raise ValueError(
            f"video bitrate too low ({bitrate:.0f} kbps, min {min_bitrate_kbps}): {path}"
        )

    video_streams = [s for s in data.get("streams", []) if s.get("codec_type") == "video"]
    if video_streams:
        vs = video_streams[0]
        avg_fps = vs.get("avg_frame_rate", "0/0")
        if avg_fps and avg_fps != "0/0":
            num, den = avg_fps.split("/")
            fps = float(num) / float(den) if float(den) else 0
            if fps > 0 and (fps < 20 or fps > 30):
                log_stage(
                    "validation",
                    message=f"unusual avg_frame_rate={avg_fps} for {path}",
                    level="ERROR",
                )
    return True


def _norm_name(value):
    return str(value or "").strip().casefold()


def validate_characters(characters, beats):
    """Ensure job has script-derived characters and beat references are valid."""
    if not characters or not isinstance(characters, list):
        raise ValueError("characters missing or empty — run script extraction")

    registry_by_name = {}
    registry_by_id = {}
    for c in characters:
        name = _norm_name(c.get("name"))
        cid = str(c.get("id") or "").strip().casefold()
        if name:
            registry_by_name[name] = c
        if cid:
            registry_by_id[cid] = c
        if not str(c.get("referencePrompt") or "").strip():
            raise ValueError(f"character missing referencePrompt: {c.get('name')}")

    for i, beat in enumerate(beats or []):
        names = beat.get("characters") or []
        if not names:
            raise ValueError(f"beat {i} has no characters assigned")
        for n in names:
            key = _norm_name(n)
            if key not in registry_by_name:
                raise ValueError(f"beat {i} unknown character: {n}")
