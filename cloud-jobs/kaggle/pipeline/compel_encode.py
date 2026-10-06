"""
SDXL long-prompt encoding via Compel (chunk past CLIP 77-token limit).

Uses CompelForSDXL (two single-encoder Compel instances) — the deprecated
list-of-tokenizers API hits EmbeddingsProviderMulti.empty_z and fails.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time

_COMPEL = None
_COMPEL_PIPE_ID = None
_COMPEL_MODE = None  # "CompelForSDXL" | "dual_compel"
_INSTALL_TRIED = False

USE_COMPEL = os.environ.get("KAGGLE_USE_COMPEL", "true").lower() not in (
    "0",
    "false",
    "no",
)

# Prefer a compel that ships CompelForSDXL (fixes empty_z on multi-encoder).
COMPEL_PIP_SPEC = os.environ.get("KAGGLE_COMPEL_SPEC", "compel>=2.3.0")


def _debug_log(hypothesis_id, location, message, data=None):
    """Print + best-effort NDJSON for local / synced workdirs."""
    payload = {
        "sessionId": "5928f0",
        "runId": "compel-fix",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data or {},
        "timestamp": int(time.time() * 1000),
    }
    print(f"   [debug:5928f0] {message} {json.dumps(data or {}, default=str)[:300]}")
    for path in (
        os.environ.get("KAGGLE_DEBUG_LOG"),
        os.path.join(os.getcwd(), "debug-5928f0.log"),
        "/kaggle/working/debug-5928f0.log",
    ):
        if not path:
            continue
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, default=str) + "\n")
            break
        except OSError:
            continue


def _pip_extra_args():
    if sys.version_info >= (3, 11):
        return ["--break-system-packages"]
    return []


def ensure_compel_installed():
    """Lazy-install/upgrade compel on Kaggle if missing or too old."""
    global _INSTALL_TRIED
    need_install = False
    try:
        import compel

        try:
            from compel import CompelForSDXL  # noqa: F401

            return True
        except ImportError:
            need_install = True
            print(
                f"   Compel {getattr(compel, '__version__', '?')} missing CompelForSDXL — upgrading…"
            )
    except ImportError:
        need_install = True

    if not need_install:
        return True
    if _INSTALL_TRIED:
        return False
    _INSTALL_TRIED = True
    print(f"📦 Installing {COMPEL_PIP_SPEC} for SDXL long-prompt embeddings…")
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-U",
            COMPEL_PIP_SPEC,
            "--no-cache-dir",
            *_pip_extra_args(),
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        print(f"   compel install failed: {(r.stderr or r.stdout or '')[-500:]}")
        return False
    try:
        from compel import CompelForSDXL  # noqa: F401

        print("   ✅ compel ready (CompelForSDXL)")
        return True
    except ImportError:
        try:
            import compel  # noqa: F401

            print("   ⚠️  compel installed but CompelForSDXL missing — using dual Compel fallback")
            return True
        except ImportError:
            return False


def sanitize_for_compel(text):
    """
    Story prompts are free-form — strip Compel weight operators so
    parentheses / + / ! do not change embedding semantics or trip SDXL pad bugs.
    """
    t = str(text or "")
    t = t.replace("!", ".")
    t = re.sub(r"[()[\]{}+]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _build_dual_compel(pipe):
    """Manual CompelForSDXL equivalent for older compel builds."""
    from compel import Compel, ReturnedEmbeddingsType

    tokenizer = getattr(pipe, "tokenizer", None)
    tokenizer_2 = getattr(pipe, "tokenizer_2", None)
    text_encoder = getattr(pipe, "text_encoder", None)
    text_encoder_2 = getattr(pipe, "text_encoder_2", None)
    if not all((tokenizer, tokenizer_2, text_encoder, text_encoder_2)):
        return None

    compel_1 = Compel(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        truncate_long_prompts=False,
    )
    compel_2 = Compel(
        tokenizer=tokenizer_2,
        text_encoder=text_encoder_2,
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        requires_pooled=True,
        truncate_long_prompts=False,
    )
    return (compel_1, compel_2)


def get_compel(pipe):
    """Cached CompelForSDXL (or dual Compel) bound to this pipeline."""
    global _COMPEL, _COMPEL_PIPE_ID, _COMPEL_MODE
    if not USE_COMPEL or pipe is None:
        return None
    if not ensure_compel_installed():
        return None

    pipe_id = id(pipe)
    if _COMPEL is not None and _COMPEL_PIPE_ID == pipe_id:
        return _COMPEL

    try:
        from compel import CompelForSDXL

        _COMPEL = CompelForSDXL(pipe)
        _COMPEL_MODE = "CompelForSDXL"
        _COMPEL_PIPE_ID = pipe_id
        _debug_log(
            "H-compel-api",
            "compel_encode.get_compel",
            "using CompelForSDXL",
            {"mode": _COMPEL_MODE},
        )
        return _COMPEL
    except Exception as e:
        _debug_log(
            "H-compel-api",
            "compel_encode.get_compel",
            "CompelForSDXL unavailable, trying dual Compel",
            {"error": str(e)[:160]},
        )

    dual = _build_dual_compel(pipe)
    if dual is None:
        print("   ⚠️  Compel: pipe missing dual CLIP encoders — falling back to trim")
        return None
    _COMPEL = dual
    _COMPEL_MODE = "dual_compel"
    _COMPEL_PIPE_ID = pipe_id
    _debug_log(
        "H-compel-api",
        "compel_encode.get_compel",
        "using dual Compel fallback",
        {"mode": _COMPEL_MODE},
    )
    return _COMPEL


def _encode_with_dual(compel_pair, pos, neg):
    """Encode with two single-encoder Compel instances (no EmbeddingsProviderMulti)."""
    import torch

    compel_1, compel_2 = compel_pair
    texts = [pos, neg if neg else ""]

    embeds_left = compel_1(texts)
    embeds_right, pooled = compel_2(texts)

    # Match CompelForSDXL padding along token dim before concat
    if embeds_left.shape[1] > embeds_right.shape[1]:
        padding, _ = compel_2([""])
        num_repeats = (embeds_left.shape[1] - embeds_right.shape[1]) // max(
            padding.shape[1], 1
        )
        if num_repeats > 0:
            embeds_right = torch.cat(
                [embeds_right, padding.repeat(embeds_right.shape[0], num_repeats, 1)],
                dim=1,
            )
    elif embeds_right.shape[1] > embeds_left.shape[1]:
        padding = compel_1([""])
        num_repeats = (embeds_right.shape[1] - embeds_left.shape[1]) // max(
            padding.shape[1], 1
        )
        if num_repeats > 0:
            embeds_left = torch.cat(
                [embeds_left, padding.repeat(embeds_left.shape[0], num_repeats, 1)],
                dim=1,
            )

    # Trim to equal length if still mismatched by a few tokens
    min_len = min(embeds_left.shape[1], embeds_right.shape[1])
    embeds_left = embeds_left[:, :min_len, :]
    embeds_right = embeds_right[:, :min_len, :]

    embeds = torch.cat([embeds_left, embeds_right], dim=-1)
    return {
        "prompt_embeds": embeds[0:1],
        "pooled_prompt_embeds": pooled[0:1],
        "negative_prompt_embeds": embeds[1:2],
        "negative_pooled_prompt_embeds": pooled[1:2],
    }


def _token_len(pipe, text):
    tok = getattr(pipe, "tokenizer", None)
    if tok is None:
        return len(str(text or "").split())
    return len(tok.encode(str(text or ""), truncation=False))


def _trim_to_tokens(pipe, text, max_tokens=70):
    """Hard-trim text to ≤max_tokens (keeps Compel on a single 77-token chunk)."""
    t = str(text or "").strip()
    if not t:
        return t
    tok = getattr(pipe, "tokenizer", None)
    if tok is None:
        return " ".join(t.split()[:max_tokens])
    ids = tok.encode(t, truncation=False)
    # CLIP specials: bos/eos usually included — leave headroom
    budget = max(8, int(max_tokens) - 2)
    if len(ids) <= budget + 2:
        return t
    trimmed = tok.decode(ids[1 : budget + 1], skip_special_tokens=True)
    return sanitize_for_compel(trimmed)


def encode_sdxl_prompts(pipe, prompt, negative_prompt=""):
    """
    Returns dict of prompt_embeds / pooled / negative embeds, or None on failure.

    Chunks long *positive* prompts via CompelForSDXL. Negatives are hard-trimmed
    to ≤70 tokens so a long neg cannot force a 154-length pad (VRAM / OOM).
    """
    compel = get_compel(pipe)
    if compel is None:
        return None

    pos = sanitize_for_compel(prompt)
    neg = sanitize_for_compel(negative_prompt or "")
    if not pos:
        return None

    # Long negatives pad BOTH sides to N×77 — trim neg, keep full positive.
    neg_raw_tokens = _token_len(pipe, neg)
    if neg_raw_tokens > 70:
        neg = _trim_to_tokens(pipe, neg, max_tokens=68)
        _debug_log(
            "H-neg-trim",
            "compel_encode.encode_sdxl_prompts",
            "trimmed long negative prompt",
            {"before": neg_raw_tokens, "after": _token_len(pipe, neg)},
        )

    try:
        if _COMPEL_MODE == "CompelForSDXL":
            # Official API: pads pos/neg internally without EmbeddingsProviderMulti.empty_z
            cond = compel(pos, negative_prompt=neg if neg else "")
            if cond.embeds is None or cond.negative_embeds is None:
                raise RuntimeError("CompelForSDXL returned empty embeds")
            result = {
                "prompt_embeds": cond.embeds,
                "pooled_prompt_embeds": cond.pooled_embeds,
                "negative_prompt_embeds": cond.negative_embeds,
                "negative_pooled_prompt_embeds": cond.negative_pooled_embeds,
            }
        else:
            result = _encode_with_dual(compel, pos, neg)

        token_est = _token_len(pipe, pos)
        neg_token_est = _token_len(pipe, neg)
        result["token_est"] = token_est
        result["neg_token_est"] = neg_token_est
        result["prompt_len_words"] = len(pos.split())
        result["mode"] = _COMPEL_MODE

        _debug_log(
            "H-compel-ok",
            "compel_encode.encode_sdxl_prompts",
            "compel encode ok",
            {
                "mode": _COMPEL_MODE,
                "words": result["prompt_len_words"],
                "token_est": token_est,
                "neg_token_est": neg_token_est,
                "embed_shape": list(result["prompt_embeds"].shape),
                "neg_shape": list(result["negative_prompt_embeds"].shape),
                "pooled_shape": list(result["pooled_prompt_embeds"].shape)
                if result["pooled_prompt_embeds"] is not None
                else None,
            },
        )
        return result
    except Exception as e:
        _debug_log(
            "H-compel-fail",
            "compel_encode.encode_sdxl_prompts",
            "compel encode failed",
            {"mode": _COMPEL_MODE, "error": str(e)[:200]},
        )
        print(f"   ⚠️  Compel encode failed ({e}) — falling back to CLIP trim")
        # Drop broken cache so next call can retry after upgrade
        global _COMPEL, _COMPEL_PIPE_ID
        _COMPEL = None
        _COMPEL_PIPE_ID = None
        return None
