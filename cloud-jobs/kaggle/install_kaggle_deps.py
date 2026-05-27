#!/usr/bin/env python3
"""
One-time Kaggle setup: pin huggingface-hub, transformers, diffusers, accelerate.

Kaggle notebook (after git bootstrap):
  !cd /kaggle/working && python cloud-jobs/kaggle/install_kaggle_deps.py

Then: Settings → Environment → Save (persists packages across sessions).
"""

import os
import sys

# Allow import from repo path or /kaggle/working
for _p in (
    "/kaggle/working",
    os.path.join(os.getcwd(), "cloud-jobs", "kaggle"),
    os.path.dirname(os.path.abspath(__file__)),
):
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from kaggle_deps import diagnose_versions, ensure_pinned_deps


def main():
    ok = ensure_pinned_deps(force=True)
    print(diagnose_versions())
    if ok:
        print(
            "\nNext: Kaggle → Settings → Environment → Save "
            "(keeps these packages for future sessions)."
        )
        print(
            "Then run: python cloud-jobs/kaggle/git_queue_processor.py once"
        )
    else:
        print(
            "\nInstall failed. Restart kernel and run this script again, "
            "or paste errors above for debugging."
        )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
