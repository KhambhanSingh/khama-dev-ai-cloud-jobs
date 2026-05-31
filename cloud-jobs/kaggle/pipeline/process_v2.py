"""Orchestrate full_video v2 pipeline."""

import os

from .logging_util import log_stage
from .validator import (
    retry_stage,
    validate_beats,
    validate_audio,
    validate_video_output,
    validate_scene_images,
    validate_characters,
)
from .audio_pipeline import build_narration_audio
from .audio_post import finalize_narration_audio, mix_bgm, resolve_bgm_path, validate_voice_present
from .image_pipeline import generate_all_scenes
from .scene_composer import compose_video
from .beat_subdivide import subdivide_long_beats


def job_work_dirs(record_id):
    rid = str(record_id)
    root = os.path.join("cloud-jobs", "work", rid)
    return {
        "root": root,
        "frames": os.path.join(root, "frames"),
        "audio": os.path.join(root, "audio"),
        "videos": os.path.join(root, "videos"),
    }


def ensure_job_dirs(d):
    for k in ("root", "frames", "audio", "videos"):
        os.makedirs(d[k], exist_ok=True)


def process_job_v2(job_data):
    record_id = job_data["recordId"]
    beats = job_data.get("beats") or []
    characters = job_data.get("characters") or []
    lang = job_data.get("language", "hi")
    video_config = job_data.get("videoConfig") or {}
    max_dur = float(video_config.get("maxDurationSec", 1800))

    if job_data.get("videoStyle"):
        video_config["videoStyle"] = job_data["videoStyle"]
    if job_data.get("narratorMode"):
        video_config["narratorMode"] = True

    validate_beats(beats)
    validate_characters(characters, beats)
    d = job_work_dirs(record_id)
    ensure_job_dirs(d)

    log_stage(
        "start",
        record_id,
        message=f"v2 beats={len(beats)} chars={len(characters)}",
    )

    def stage_tts():
        return build_narration_audio(
            record_id, beats, lang, d["audio"], max_duration_sec=max_dur
        )

    audio_path, timings, _caption_words, total_audio = retry_stage(
        stage_tts, "tts", record_id
    )
    validate_audio(audio_path)

    bgm_path = resolve_bgm_path(job_data)
    if bgm_path:
        log_stage("tts", record_id, message=f"mixing bgm {os.path.basename(bgm_path)}")
        mix_bgm(audio_path, bgm_path, audio_path)
    else:
        finalize_narration_audio(audio_path, audio_path)

    validate_voice_present(audio_path)
    log_stage("tts", record_id, message=f"tts_done duration={total_audio:.1f}s")

    for beat in beats:
        idx = beat.get("sceneIndex")
        for t in timings:
            if t["sceneIndex"] == idx:
                beat["duration"] = t["duration"]
                break

    beats, timings = subdivide_long_beats(beats, timings, max_sec=4.5)
    log_stage(
        "start",
        record_id,
        message=f"after_subdivide beats={len(beats)} clips≈{len(timings)}",
    )

    def stage_images():
        return generate_all_scenes(
            record_id, beats, characters, video_config, d
        )

    scene_images = retry_stage(stage_images, "image", record_id)
    validate_scene_images(scene_images, audio_duration_sec=total_audio)
    log_stage("image", record_id, message="images_done")

    compose_attempt = [0]

    def stage_compose():
        compose_attempt[0] += 1
        return compose_video(
            record_id,
            scene_images,
            timings,
            audio_path,
            d,
            video_config,
            reuse_clips=compose_attempt[0] > 1,
        )

    raw_video = retry_stage(stage_compose, "compose", record_id)

    # Captions/subtitles feature removed — raw_video is the final output.
    final_video = raw_video
    validate_video_output(final_video)
    log_stage("validation", record_id, message="validation_pass")

    log_stage("done", record_id, message=f"duration={total_audio:.1f}s")

    return {
        "recordId": record_id,
        "video": final_video,
        "audio": audio_path,
        "audioDuration": total_audio,
        "captionJson": [],
        "beatTimings": timings,
    }
