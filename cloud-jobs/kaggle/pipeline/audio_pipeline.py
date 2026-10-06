"""Per-beat TTS, silence trim, normalize, concat."""

import os
import re
import tempfile

from gtts import gTTS
from pydub import AudioSegment
from pydub.effects import speedup, normalize

from .logging_util import log_stage

# Emotion → playback factor (speed) and gain dB
EMOTION_AUDIO = {
    "excited": {"speed": 1.08, "gain": 4},
    "happy": {"speed": 1.05, "gain": 2},
    "angry": {"speed": 0.98, "gain": 5},
    "sad": {"speed": 0.92, "gain": -3},
    "scared": {"speed": 1.06, "gain": 0},
    "panic": {"speed": 1.12, "gain": 2},
    "suspense": {"speed": 0.95, "gain": -2},
    "sleepy": {"speed": 0.9, "gain": -2},
    "asleep": {"speed": 0.88, "gain": -3},
    "thinking": {"speed": 0.96, "gain": 0},
    "laughing": {"speed": 1.1, "gain": 2},
    "crying": {"speed": 0.9, "gain": -2},
    "neutral": {"speed": 1.0, "gain": 0},
    "calm": {"speed": 0.98, "gain": 0},
    "relaxed": {"speed": 0.97, "gain": 0},
}

SILENCE_THRESH_DB = -40
MIN_SILENCE_LEN_MS = 250
MAX_CHUNK_CHARS = 4500


def _split_sentences(text):
    parts = re.split(r"(?<=[।.!?])\s+", str(text or "").strip())
    return [p.strip() for p in parts if p.strip()]


def _chunk_text(text, max_chars=MAX_CHUNK_CHARS):
    text = str(text or "").strip()
    if len(text) <= max_chars:
        return [text] if text else []
    chunks = []
    buf = ""
    for sent in _split_sentences(text):
        if len(buf) + len(sent) + 1 > max_chars and buf:
            chunks.append(buf.strip())
            buf = sent
        else:
            buf = f"{buf} {sent}".strip() if buf else sent
    if buf:
        chunks.append(buf.strip())
    return chunks


def _trim_silence(audio, silence_thresh=SILENCE_THRESH_DB, min_silence_len=MIN_SILENCE_LEN_MS):
    from pydub.silence import split_on_silence

    chunks = split_on_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=80,
    )
    if not chunks:
        return audio
    out = AudioSegment.empty()
    for c in chunks:
        out += c
    return out


def _apply_emotion(audio, emotion):
    cfg = EMOTION_AUDIO.get(str(emotion or "neutral").lower(), EMOTION_AUDIO["neutral"])
    spd = cfg.get("speed", 1.0)
    if spd > 1.01:
        audio = speedup(audio, playback_speed=spd, chunk_size=150, crossfade=25)
    elif spd < 0.99:
        # slow down by frame rate manipulation
        audio = audio._spawn(
            audio.raw_data, overrides={"frame_rate": int(audio.frame_rate * spd)}
        ).set_frame_rate(audio.frame_rate)
    gain = cfg.get("gain", 0)
    if gain:
        audio = audio + gain
    return audio


def synthesize_beat_tts(text, lang, out_wav, emotion="neutral"):
    chunks = _chunk_text(text)
    if not chunks:
        raise ValueError("empty narration for TTS")

    combined = AudioSegment.empty()
    tmpdir = tempfile.mkdtemp(prefix="beat_tts_")

    try:
        for i, chunk in enumerate(chunks):
            mp3 = os.path.join(tmpdir, f"c_{i}.mp3")
            gTTS(text=chunk, lang=lang or "hi", slow=False).save(mp3)
            seg = AudioSegment.from_file(mp3)
            combined += seg

        combined = _trim_silence(combined)
        combined = _apply_emotion(combined, emotion)
        combined = normalize(combined)
        combined.export(out_wav, format="wav")
        wav_size = os.path.getsize(out_wav)
        if wav_size < 5_000:
            raise ValueError(
                f"TTS WAV too small ({wav_size} bytes) — gTTS likely failed or produced silence"
            )
        log_stage(
            "tts",
            message=f"beat wav size={wav_size} dur={len(combined)/1000.0:.1f}s path={out_wav}",
        )
        return len(combined) / 1000.0
    finally:
        try:
            for f in os.listdir(tmpdir):
                os.remove(os.path.join(tmpdir, f))
            os.rmdir(tmpdir)
        except OSError:
            pass


def build_narration_audio(record_id, beats, lang, audio_dir, max_duration_sec=1800):
    """
    Returns (master_wav_path, beat_timings, total_sec).
    beat_timings: [{ sceneIndex, start, end, duration, narrationText }]
    """
    os.makedirs(audio_dir, exist_ok=True)
    master = AudioSegment.empty()
    timings = []
    cursor = 0.0

    for beat in beats:
        idx = beat.get("sceneIndex", len(timings))
        text = str(beat.get("narrationText", "")).strip()
        emotion = beat.get("emotion", "neutral")
        wav_path = os.path.join(audio_dir, f"beat_{idx:03d}.wav")

        log_stage("tts", record_id, beat=idx, message=f"chars={len(text)}")

        if not text:
            continue

        duration = synthesize_beat_tts(text, lang, wav_path, emotion)
        seg = AudioSegment.from_wav(wav_path)
        master += seg

        start = cursor
        end = cursor + duration
        cursor = end

        timings.append(
            {
                "sceneIndex": idx,
                "start": start,
                "end": end,
                "duration": duration,
                "narrationText": text,
                "action": str(beat.get("action", "")).strip(),
                "actionPose": str(beat.get("actionPose") or beat.get("action", "")).strip(),
                "cameraStyle": str(beat.get("cameraStyle", "")).strip(),
                "emotion": str(beat.get("emotion", "neutral")).strip(),
            }
        )

    if not timings:
        raise ValueError("no audio segments generated")

    total = len(master) / 1000.0
    if total > max_duration_sec:
        ratio = max_duration_sec / total
        log_stage(
            "tts",
            record_id,
            message=f"scale audio {total:.1f}s -> {max_duration_sec}s ratio={ratio:.3f}",
        )
        master = master[: int(max_duration_sec * 1000)]
        total = len(master) / 1000.0
        cursor = 0.0
        for t in timings:
            t["duration"] *= ratio
            t["start"] = cursor
            t["end"] = cursor + t["duration"]
            cursor = t["end"]

    master_path = os.path.join(audio_dir, f"{record_id}.wav")
    master = normalize(master)
    # 500 ms safety buffer so audio is never shorter than the composed video (BUG 8)
    master += AudioSegment.silent(duration=500)
    master.export(master_path, format="wav")
    log_stage(
        "tts",
        record_id,
        message=f"master_wav size={os.path.getsize(master_path)} dur={total:.1f}s path={master_path}",
    )

    return master_path, timings, total
