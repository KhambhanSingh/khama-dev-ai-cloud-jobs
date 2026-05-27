"""FFmpeg scene clips with zoompan and concat + transitions."""

import os
import subprocess

from .logging_util import log_stage

from .validator import validate_scene_clips, validate_scene_png, validate_video


def _run_ffmpeg(cmd):
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        err = (e.stderr or b"") + (e.stdout or b"")
        err = err.decode("utf-8", errors="replace")[:2000]
        log_stage("compose", message=err, level="ERROR")
        raise


def _video_encode_args(fps):
    fps = max(1, int(fps))
    return [
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-b:v",
        "2500k",
        "-maxrate",
        "4000k",
        "-bufsize",
        "8000k",
        "-pix_fmt",
        "yuv420p",
        "-vsync",
        "cfr",
        "-r",
        str(fps),
        "-g",
        str(fps),
        "-keyint_min",
        str(fps),
        "-movflags",
        "+faststart",
    ]


def _zoompan_vf(width, height, fps, duration_sec):
    duration_sec = max(0.5, float(duration_sec))
    frames = max(2, int(duration_sec * fps))
    return (
        f"scale={width}:{height},"
        f"zoompan=z='min(zoom+0.001,1.12)':d={frames}:"
        f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={width}x{height}:fps={fps}"
    )


def _encode_scene_clip(image_path, out_path, duration_sec, width, height, fps):
    """Encode one scene clip with fixed params for concat compatibility."""
    duration_sec = max(0.5, float(duration_sec))
    fps = max(1, int(fps))
    cmd = [
        "ffmpeg",
        "-y",
        "-loop",
        "1",
        "-i",
        image_path,
        "-vf",
        _zoompan_vf(width, height, fps, duration_sec),
        "-t",
        str(duration_sec),
    ] + _video_encode_args(fps) + [out_path]
    _run_ffmpeg(cmd)
    return out_path


def render_scene_clip(image_path, out_path, duration_sec, fps=24, width=1280, height=720):
    """Ken Burns zoom on still image for exact duration."""
    return _encode_scene_clip(image_path, out_path, duration_sec, width, height, fps)


def _write_concat_list(list_path, clip_paths):
    lines = []
    for p in clip_paths:
        ap = os.path.abspath(p)
        if not os.path.isfile(ap):
            raise FileNotFoundError(f"missing clip: {ap}")
        safe = ap.replace("'", "'\\''")
        lines.append(f"file '{safe}'")
    with open(list_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _concat_copy(list_path, video_only, fps):
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-fflags",
        "+genpts",
        "-i",
        list_path,
        "-c",
        "copy",
        "-vsync",
        "cfr",
        "-r",
        str(fps),
        video_only,
    ]
    _run_ffmpeg(cmd)


def _concat_reencode(list_path, video_only, fps):
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-fflags",
        "+genpts",
        "-i",
        list_path,
    ] + _video_encode_args(fps) + [video_only]
    _run_ffmpeg(cmd)


def concat_scenes_with_audio(
    record_id,
    clip_paths,
    audio_path,
    out_path,
    fps=24,
    fade_sec=0.35,
):
    if not clip_paths:
        raise ValueError("no scene clips")

    work = os.path.dirname(os.path.abspath(clip_paths[0]))
    list_path = os.path.join(work, "concat_list.txt")
    fps = max(1, int(fps))

    if len(clip_paths) == 1:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            os.path.abspath(clip_paths[0]),
            "-i",
            audio_path,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-ar",
            "44100",
            "-ac",
            "2",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-vsync",
            "cfr",
            "-r",
            str(fps),
            "-movflags",
            "+faststart",
            "-shortest",
            out_path,
        ]
        _run_ffmpeg(cmd)
        validate_video(out_path)
        return out_path

    _write_concat_list(list_path, clip_paths)
    video_only = os.path.join(work, f"{record_id}_video_only.mp4")

    try:
        _concat_copy(list_path, video_only, fps)
    except subprocess.CalledProcessError:
        log_stage(
            "compose",
            record_id,
            message="concat copy failed, trying re-encode fallback",
        )
        _concat_reencode(list_path, video_only, fps)

    cmd_mux = [
        "ffmpeg",
        "-y",
        "-i",
        video_only,
        "-i",
        audio_path,
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-ar",
        "44100",
        "-ac",
        "2",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-vsync",
        "cfr",
        "-r",
        str(fps),
        "-movflags",
        "+faststart",
        "-shortest",
        out_path,
    ]
    _run_ffmpeg(cmd_mux)
    validate_video(out_path)
    return out_path


def compose_video(
    record_id,
    scene_images,
    beat_timings,
    audio_path,
    work_dirs,
    video_config,
    reuse_clips=False,
):
    fps = int(video_config.get("fps", 24))
    width = int(video_config.get("width", 1280))
    height = int(video_config.get("height", 720))
    clips_dir = os.path.join(work_dirs["root"], "clips")
    os.makedirs(clips_dir, exist_ok=True)

    timing_by_idx = {t["sceneIndex"]: t for t in beat_timings}
    sorted_imgs = sorted(
        scene_images,
        key=lambda p: int(
            os.path.basename(p).replace("scene_", "").replace(".png", "")
        ),
    )

    total_video_sec = sum(
        float(timing_by_idx.get(
            int(os.path.basename(img).replace("scene_", "").replace(".png", "")),
            {},
        ).get("duration", 3.0))
        for img in sorted_imgs
    )
    log_stage(
        "compose",
        record_id,
        message=f"clip_count={len(sorted_imgs)} total_video_sec≈{total_video_sec:.1f}",
    )

    clip_paths = []

    for img in sorted_imgs:
        validate_scene_png(img)
        base = os.path.basename(img)
        idx = int(base.replace("scene_", "").replace(".png", ""))
        dur = float(timing_by_idx.get(idx, {}).get("duration", 3.0))
        clip_out = os.path.join(clips_dir, f"clip_{idx:03d}.mp4")

        if reuse_clips and os.path.isfile(clip_out):
            log_stage("compose", record_id, beat=idx, message="reuse existing clip")
            clip_paths.append(clip_out)
            continue

        log_stage("compose", record_id, beat=idx, message=f"duration={dur:.2f}s")
        _encode_scene_clip(img, clip_out, dur, width, height, fps)
        clip_paths.append(clip_out)

    validate_scene_clips(clip_paths)
    videos_dir = work_dirs["videos"]
    os.makedirs(videos_dir, exist_ok=True)
    final = os.path.join(videos_dir, f"{record_id}.mp4")
    concat_scenes_with_audio(record_id, clip_paths, audio_path, final, fps=fps)
    log_stage("compose", record_id, message="compose_done")
    return final
