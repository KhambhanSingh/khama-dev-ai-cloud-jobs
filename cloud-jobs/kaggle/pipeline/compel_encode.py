"""
SDXL long-prompt encoding via Compel (chunk past CLIP 77-token limit).

Uses prompt_embeds / pooled_prompt_embeds so scene + character details are not
silently truncated by the CLIP tokenizer.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

_COMPEL = None
_COMPEL_PIPE_ID = None
_INSTALL_TRIED = False

USE_COMPEL = os.environ.get("KAGGLE_USE_COMPEL", "true").lower() not in (
    "0",
    "false",
    "no",
)


def _pip_extra_args():
    if sys.version_info >= (3, 11):
        return ["--break-system-packages"]
    return []


def ensure_compel_installed():
    """Lazy-install compel on Kaggle if missing."""
    global _INSTALL_TRIED
    try:
        import compel  # noqa: F401

        return True
    except ImportError:
        pass
    if _INSTALL_TRIED:
        return False
    _INSTALL_TRIED = True
    print("📦 Installing compel for SDXL long-prompt embeddings…")
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-U",
            "compel",
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
        import compel  # noqa: F401

        print("   ✅ compel ready")
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


def get_compel(pipe):
    """Cached Compel bound to this pipeline's dual CLIP encoders."""
    global _COMPEL, _COMPEL_PIPE_ID
    if not USE_COMPEL or pipe is None:
        return None
    if not ensure_compel_installed():
        return None

    pipe_id = id(pipe)
    if _COMPEL is not None and _COMPEL_PIPE_ID == pipe_id:
        return _COMPEL

    from compel import Compel, ReturnedEmbeddingsType

    tokenizer = getattr(pipe, "tokenizer", None)
    tokenizer_2 = getattr(pipe, "tokenizer_2", None)
    text_encoder = getattr(pipe, "text_encoder", None)
    text_encoder_2 = getattr(pipe, "text_encoder_2", None)
    if not all((tokenizer, tokenizer_2, text_encoder, text_encoder_2)):
        print("   ⚠️  Compel: pipe missing dual CLIP encoders — falling back to trim")
        return None

    _COMPEL = Compel(
        tokenizer=[tokenizer, tokenizer_2],
        text_encoder=[text_encoder, text_encoder_2],
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        requires_pooled=[False, True],
        truncate_long_prompts=False,
    )
    _COMPEL_PIPE_ID = pipe_id
    return _COMPEL


def encode_sdxl_prompts(pipe, prompt, negative_prompt=""):
    """
    Returns dict of prompt_embeds / pooled / negative embeds, or None on failure.

    Chunks prompts longer than 77 tokens; pads pos/neg to matching sequence length.
    """
    compel = get_compel(pipe)
    if compel is None:
        return None

    pos = sanitize_for_compel(prompt)
    neg = sanitize_for_compel(negative_prompt or "")
    if not pos:
        return None

    try:
        # Batch encode so Compel pads chunk counts consistently when possible
        conditioning, pooled = compel([pos, neg if neg else ""])
        prompt_embeds = conditioning[0:1]
        negative_prompt_embeds = conditioning[1:2]
        pooled_prompt_embeds = pooled[0:1]
        negative_pooled_prompt_embeds = pooled[1:2]

        # Extra safety if lengths still diverge
        if prompt_embeds.shape[1] != negative_prompt_embeds.shape[1]:
            prompt_embeds, negative_prompt_embeds = compel.pad_conditioning_tensors_to_same_length(
                [prompt_embeds.squeeze(0), negative_prompt_embeds.squeeze(0)]
            )
            if prompt_embeds.dim() == 2:
                prompt_embeds = prompt_embeds.unsqueeze(0)
            if negative_prompt_embeds.dim() == 2:
                negative_prompt_embeds = negative_prompt_embeds.unsqueeze(0)

        # Rough token estimate for logs (tokenizer_1, no truncate)
        tok = getattr(pipe, "tokenizer", None)
        token_est = (
            len(tok.encode(pos, truncation=False)) if tok is not None else len(pos.split())
        )
        return {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
            "token_est": token_est,
            "prompt_len_words": len(pos.split()),
        }
    except Exception as e:
        print(f"   ⚠️  Compel encode failed ({e}) — falling back to CLIP trim")
        return None
