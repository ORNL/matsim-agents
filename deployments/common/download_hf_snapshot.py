"""Download a Hugging Face repository directly into a durable local directory."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import urllib.request

from huggingface_hub import HfApi, hf_hub_url


def download_file(repo_id: str, filename: str, destination: Path, token: str | None) -> None:
    if destination.is_file() and destination.stat().st_size:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = hf_hub_url(repo_id, filename)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    request = urllib.request.Request(url, headers=headers)
    temporary = destination.with_suffix(destination.suffix + f".part.{os.getpid()}")
    try:
        with urllib.request.urlopen(request) as response, temporary.open("wb") as output:
            while chunk := response.read(8 * 1024 * 1024):
                output.write(chunk)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id")
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    api = HfApi(token=token)
    files = api.list_repo_files(args.repo_id, repo_type="model", token=token)
    for filename in files:
        print(f"downloading {args.repo_id}/{filename}", flush=True)
        download_file(args.repo_id, filename, args.destination / filename, token)


if __name__ == "__main__":
    main()