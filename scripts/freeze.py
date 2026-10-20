"""Step 4: freeze the configuration and prove it was not changed.

  python scripts/freeze.py lock   config/frozen_config.json   -> writes config/frozen.lock
  python scripts/freeze.py verify config/frozen_config.json   -> fails if any file changed

The lock stores a SHA-256 hash of the config file and of every prompt file it
references. Run `verify` before each batch of model calls.
Prompt paths in the config are relative to the directory you run from (the
project root). Lock keys are stored relative to the config file's folder, so
`config/x.json`, `./config/x.json` and an absolute path all verify alike.
"""
import hashlib
import json
import os
import sys
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def files_of(config_path):
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    return [config_path, *cfg.get("prompts", {}).values()]


def key_of(path, config_path):
    base = Path(config_path).resolve().parent
    return Path(os.path.relpath(Path(path).resolve(), base)).as_posix()


def main(cmd, config_path):
    lock = Path(config_path).with_name("frozen.lock")
    current = {key_of(f, config_path): digest(f) for f in files_of(config_path)}
    if cmd == "lock":
        lock.write_text(json.dumps(current, indent=2), encoding="utf-8")
        print(f"locked {len(current)} files -> {lock}")
    elif cmd == "verify":
        saved = json.loads(lock.read_text(encoding="utf-8"))
        changed = [f for f in saved.keys() | current.keys() if saved.get(f) != current.get(f)]
        if changed:
            sys.exit(f"CHANGED since freeze: {changed}. This is a new experiment: re-number it.")
        print("configuration unchanged since freeze")
    else:
        sys.exit("usage: freeze.py lock|verify CONFIG")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
