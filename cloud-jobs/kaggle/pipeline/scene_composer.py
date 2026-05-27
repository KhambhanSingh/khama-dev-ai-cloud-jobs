"""FFmpeg scene clips with zoompan and concat + transitions."""

import os
import subprocess

from .logging_util import log_stage

from .validator import validate_scene_clips, validate_video


def _run_ffmpeg(cmd):
    subprocess.run(cmd, check=True, capture_output=True)


def render_scene_clip(image_path, out_path, duration_sec, fps=24):
    """Ken Burns zoom on still image for exact duration."""
    duration_sec = max(0.5, float(duration_sec))
    frames = max(2, int(duration_sec * fps))
    vf = (
        f"zoompan=z='min(zoom+0.001,1.12)':d={frames}:"
        f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1280x720:fps={fps}"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-loop",
        "1",
        "-i",
        image_path,
        "-vf",
        vf,
        "-t",
        str(duration_sec),
        "-r",
        str(fps),
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        out_path,
    ]
    _run_ffmpeg(cmd)
    return out_path


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

    work = os.path.dirname(clip_paths[0])
    list_path = os.path.join(work, "concat_list.txt")

    # Simple concat demuxer then mux audio
    if len(clip_paths) == 1:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            clip_paths[0],
            "-i",
            audio_path,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            out_path,
        ]
        _run_ffmpeg(cmd)
        return out_path

    # Concat demuxer then mux audio (video length must match audio)
    with open(list_path, "w", encoding="utf-8") as f:
        for p in clip_paths:
            safe = p.replace("'", "'\\''")
            f.write(f"file '{safe}'\n")

    video_only = os.path.join(work, f"{record_id}_video_only.mp4")
    cmd_concat = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        list_path,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        video_only,
    ]
    _run_ffmpeg(cmd_concat)

    cmd_mux = [
        "ffmpeg",
        "-y",
        "-i",
        video_only,
        "-i",
        audio_path,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
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
):
    fps = int(video_config.get("fps", 24))
    width = int(video_config.get("width", 1280))
    height = int(video_config.get("height", 720))
    clips_dir = os.path.join(work_dirs["root"], "clips")
    os.makedirs(clips_dir, exist_ok=True)

    timing_by_idx = {t["sceneIndex"]: t for t in beat_timings}
    clip_paths = []

    for img in scene_images:
        base = os.path.basename(img)
        idx = int(base.replace("scene_", "").replace(".png", ""))
        dur = timing_by_idx.get(idx, {}).get("duration", 3.0)
        clip_out = os.path.join(clips_dir, f"clip_{idx:03d}.mp4")
        log_stage("compose", record_id, beat=idx, message=f"duration={dur:.2f}s")

        # Scale in ffmpeg if not 1280x720
        vf_scale = f"scale={width}:{height}"
        duration_sec = max(0.5, float(dur))
        frames = max(2, int(duration_sec * fps))
        vf = (
            f"{vf_scale},zoompan=z='min(zoom+0.001,1.12)':d={frames}:"
            f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={width}x{height}:fps={fps}"
        )
        cmd = [
            "ffmpeg",
            "-y",
            "-loop",
            "1",
            "-i",
            img,
            "-vf",
            vf,
            "-t",
            str(duration_sec),
            "-r",
            str(fps),
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            clip_out,
        ]
        subprocess.run(cmd, check=True)
        clip_paths.append(clip_out)

    validate_scene_clips(clip_paths)
    videos_dir = work_dirs["videos"]
    os.makedirs(videos_dir, exist_ok=True)
    final = os.path.join(videos_dir, f"{record_id}.mp4")
    concat_scenes_with_audio(record_id, clip_paths, audio_path, final, fps=fps)
    return final
