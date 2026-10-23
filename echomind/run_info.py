import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

# saved next to every result, so a number in the report can be traced back
# to the exact code, data and package versions that produced it

PACKAGES = [
    "faster-whisper", "ctranslate2", "av", "jiwer", "torch", "transformers",
    "peft", "accelerate", "datasets", "librosa", "pyannote.audio",
    "speechbrain",
]


REPO_ROOT = Path(__file__).resolve().parent.parent


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=True, cwd=REPO_ROOT)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                               capture_output=True, text=True, check=True)
        commit = out.stdout.strip()
        return commit + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def package_versions():
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return versions


def gpu_name():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return None


def run_info(settings, manifest_path=None):
    info = {
        "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "gpu": gpu_name(),
        "packages": package_versions(),
        "settings": settings,
    }
    if manifest_path and Path(manifest_path).exists():
        info["manifest_sha256"] = file_sha256(manifest_path)
    return info
