"""SRT generation and ffmpeg subtitle burn-in."""

import os
import subprocess

from .logging_util import log_stage

# Map UI caption style names to ASS force_style.
# NOTE: ASS colours are &HAABBGGRR (BGR byte order), NOT RGB.
#   White  = &H00FFFFFF  (B=FF G=FF R=FF) — correct
#   Black  = &H00000000  — correct
#   Gold   = &H0000D7FF  (B=00 G=D7 R=FF — gold in BGR, not &H00FFD700 which is cyan)
#   Orange = &H000066FF  (B=00 G=66 R=FF)
#   Green  = &H0000FF00  (B=00 G=FF R=00) — correct
#   Purple = &H00FF66FF  (B=FF G=66 R=FF) — correct
# BUG 9: All styles now use correct BGR colors + white+black-outline baseline for readability.
# BUG 6: MarginL=80,MarginR=80 and WrapStyle=0 added to all styles to prevent overflow.
CAPTION_STYLES = {
    "Youtuber": (
        "FontName=Arial,FontSize=22,Bold=1,"
        "PrimaryColour=&H00FFFFFF,"        # white (universally readable)
        "OutlineColour=&H00000000,"        # black outline
        "BackColour=&H80000000,"           # semi-transparent black shadow box
        "BorderStyle=1,Outline=3,Shadow=1,Alignment=2,"
        "MarginV=40,MarginL=80,MarginR=80,WrapStyle=0"
    ),
    "Caption": (
        "FontName=Arial,FontSize=20,Bold=1,"
        "PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,"
        "BorderStyle=1,Outline=2,Shadow=1,Alignment=2,"
        "MarginV=36,MarginL=80,MarginR=80,WrapStyle=0"
    ),
    "Kids": (
        "FontName=Arial,FontSize=28,Bold=1,"
        "PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H000066FF,"        # orange outline in BGR: B=00 G=66 R=FF
        "BorderStyle=1,Outline=3,Shadow=1,Alignment=2,"
        "MarginV=40,MarginL=80,MarginR=80,WrapStyle=0"
    ),
    "Minimal": (
        "FontName=Arial,FontSize=18,Bold=0,"
        "PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,"
        "BorderStyle=1,Outline=1,Alignment=2,"
        "MarginV=28,MarginL=80,MarginR=80,WrapStyle=0"
    ),
    "Neon": (
        "FontName=Arial,FontSize=22,Bold=1,"
        "PrimaryColour=&H0000FF00,"        # green — BGR &H0000FF00 is correct green
        "OutlineColour=&H00000000,"
        "BorderStyle=1,Outline=2,Alignment=2,"
        "MarginV=36,MarginL=80,MarginR=80,WrapStyle=0"
    ),
    "Supreme": (
        "FontName=Arial,FontSize=20,Bold=1,Italic=1,"
        "PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,"
        "BorderStyle=1,Outline=2,Shadow=1,Alignment=2,"
        "MarginV=36,MarginL=80,MarginR=80,WrapStyle=0"
    ),
    "Gradient": (
        "FontName=Arial,FontSize=20,Bold=1,"
        "PrimaryColour=&H0000D7FF,"        # gold in BGR: B=00 G=D7 R=FF
        "OutlineColour=&H00000000,"
        "BorderStyle=1,Outline=2,Shadow=1,Alignment=2,"
        "MarginV=36,MarginL=80,MarginR=80,WrapStyle=0"
    ),
    "Glitch": (
        "FontName=Arial,FontSize=22,Bold=1,"
        "PrimaryColour=&H00FF66FF,"        # purple: B=FF G=66 R=FF — BGR correct
        "OutlineColour=&H00000000,"
        "BorderStyle=1,Outline=2,Alignment=2,"
        "MarginV=36,MarginL=80,MarginR=80,WrapStyle=0"
    ),
}


def _sec_to_srt(t):
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int((t % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _group_phrases(caption_words, max_words=6, max_gap=0.6):
    if not caption_words:
        return []
    phrases = []
    buf = []
    start = caption_words[0]["start"]

    for i, c in enumerate(caption_words):
        buf.append(c)
        is_last = i == len(caption_words) - 1
        gap = (
            caption_words[i + 1]["start"] - c["end"]
            if not is_last
            else 0
        )
        if len(buf) >= max_words or gap > max_gap or is_last:
            phrases.append(
                {
                    "text": " ".join(x["word"] for x in buf),
                    "start": start,
                    "end": buf[-1]["end"],
                }
            )
            buf = []
            if not is_last:
                start = caption_words[i + 1]["start"]
    return phrases


def write_srt(caption_words, srt_path):
    phrases = _group_phrases(caption_words)
    lines = []
    for i, p in enumerate(phrases, 1):
        lines.append(str(i))
        lines.append(f"{_sec_to_srt(p['start'])} --> {_sec_to_srt(p['end'])}")
        lines.append(p["text"])
        lines.append("")
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return srt_path


def burn_captions(record_id, input_video, caption_words, caption_style, work_dirs):
    srt_path = os.path.join(work_dirs["root"], "captions.srt")
    write_srt(caption_words, srt_path)

    if not caption_words or not os.path.isfile(srt_path):
        log_stage("captions", record_id, message="skip burn — no captions")
        return input_video

    style_name = "Youtuber"
    if isinstance(caption_style, dict):
        style_name = caption_style.get("name") or style_name
    elif isinstance(caption_style, str):
        style_name = caption_style

    force_style = CAPTION_STYLES.get(style_name, CAPTION_STYLES["Youtuber"])
    out_path = os.path.join(work_dirs["videos"], f"{record_id}_captioned.mp4")

    sub_path = srt_path.replace("\\", "/")
    if os.name == "nt" and len(sub_path) > 1 and sub_path[1] == ":":
        sub_path = sub_path[0] + "\\:" + sub_path[2:]

    vf = f"subtitles='{sub_path}':force_style='{force_style}'"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_video,
        "-vf",
        vf,
        "-c:a",
        "copy",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        out_path,
    ]
    log_stage("captions", record_id, message=f"style={style_name}")
    subprocess.run(cmd, check=True)
    return out_path
