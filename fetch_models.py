"""Retrieve pinned public model files and verify every byte before installation."""
import hashlib
import json
import os
import urllib.request
from pathlib import Path

root = Path(__file__).resolve().parent
for model in json.loads((root / "MODEL_MANIFEST.json").read_text()):
    target = root / "models" / model["filename"]
    if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == model["sha256"]:
        print("Verified", model["filename"])
        continue
    target.parent.mkdir(exist_ok=True)
    temporary = target.with_suffix(".download")
    request = urllib.request.Request(model["url"], headers={"User-Agent": "MargAI-submission-reproduction"})
    try:
        digest = hashlib.sha256()
        size = 0
        with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        if size != model["bytes"] or digest.hexdigest() != model["sha256"]:
            raise RuntimeError("Model integrity check failed: " + model["filename"])
        os.replace(temporary, target)
        print("Downloaded and verified", model["filename"])
    finally:
        temporary.unlink(missing_ok=True)
