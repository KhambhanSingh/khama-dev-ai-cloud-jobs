"""
Clone or pull the cloud-jobs GitHub repo into /kaggle/working before running
git_queue_processor.py.

Kaggle notebook:
  !python cloud-jobs/kaggle/kaggle_bootstrap.py

Requires Kaggle secrets: GITHUB_TOKEN, NEXT_GITHUB_REPO
"""

import os
import subprocess
import sys

WORKING = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
BRANCH = os.environ.get("GIT_BRANCH", "main").strip() or "main"
WORKER_PATH = "cloud-jobs/kaggle/git_queue_processor.py"


def load_secrets():
    try:
        from kaggle_secrets import UserSecretsClient

        s = UserSecretsClient()
        token = s.get_secret("GITHUB_TOKEN")
        repo = s.get_secret("NEXT_GITHUB_REPO")
    except Exception as e:
        print(f"Failed to load Kaggle secrets: {e}")
        print("Set GITHUB_TOKEN and NEXT_GITHUB_REPO in Add-ons → Secrets.")
        sys.exit(1)

    if not token or not repo:
        print("GITHUB_TOKEN and NEXT_GITHUB_REPO must be set in Kaggle secrets.")
        sys.exit(1)
    return token, repo.strip()


def _init_and_fetch(auth):
    """Sync into non-empty /kaggle/working (git clone . would fail)."""
    print(f"Initializing git in {WORKING} and syncing ({BRANCH}) ...")
    subprocess.run(["git", "init"], check=True)
    subprocess.run(["git", "checkout", "-b", BRANCH], check=False)
    subprocess.run(["git", "symbolic-ref", "HEAD", f"refs/heads/{BRANCH}"], check=True)
    subprocess.run(["git", "remote", "add", "origin", auth], check=False)
    subprocess.run(["git", "remote", "set-url", "origin", auth], check=True)
    subprocess.run(["git", "fetch", "origin", BRANCH], check=True)
    subprocess.run(["git", "reset", "--hard", f"origin/{BRANCH}"], check=True)


def sync_repo(token, repo):
    os.chdir(WORKING)
    auth = f"https://x-access-token:{token}@github.com/{repo}.git"

    if os.path.exists(".git"):
        print(f"Pulling latest {repo} ({BRANCH}) ...")
        subprocess.run(["git", "remote", "set-url", "origin", auth], check=True)
        subprocess.run(["git", "fetch", "origin", BRANCH], check=True)
        subprocess.run(["git", "reset", "--hard", f"origin/{BRANCH}"], check=True)
    else:
        _init_and_fetch(auth)


def verify_workers():
    if not os.path.isfile(WORKER_PATH):
        print(
            f"Missing {WORKER_PATH}. Generate one full_video from the app "
            "so workers are pushed to GitHub, then run this script again."
        )
        sys.exit(1)


def main():
    token, repo = load_secrets()
    sync_repo(token, repo)
    verify_workers()
    print("OK — repo synced")
    print(f"Next: python {WORKER_PATH} once")


if __name__ == "__main__":
    main()
