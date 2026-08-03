# -*- coding: utf-8 -*-
"""Normalize the released checkpoints' path metadata.

Ultralytics records the training invocation inside the checkpoint, so a
finetune produced on a workstation carries that machine's absolute paths in
`train_args` (model / data / project / save_dir / name). Those fields are
useless to anyone else -- they point at directories that do not exist on their
machine -- and they publish the author's local layout.

This rewrites only those metadata strings. Tensors, EMA buffers, class names,
and every field inference reads are left untouched, and the script proves that
by comparing every tensor before and after byte for byte.

    python scripts/sanitize_checkpoints.py --check    # report only
    python scripts/sanitize_checkpoints.py --apply    # rewrite in place
"""
import argparse
import hashlib
import io
import os
import sys

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.environ.get("DART_ROOT",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WDIR = os.path.join(ROOT, "weights_release")

# Repository-relative replacements, keyed by the dataset each checkpoint used.
DATA_CFG = {
    "visdrone_yolo26n_1280.pt": "configs/visdrone.yaml",
    "visdrone_yolo26s_1280.pt": "configs/visdrone.yaml",
    "sku110k_yolo26n_1024.pt": "configs/sku110k.yaml",
}
BASE = {
    "visdrone_yolo26n_1280.pt": "yolo26n.pt",
    "visdrone_yolo26s_1280.pt": "yolo26s.pt",
    "sku110k_yolo26n_1024.pt": "yolo26n.pt",
}
PATH_KEYS = ("model", "data", "project", "save_dir", "name", "source", "weights")


def tensor_digest(obj):
    """Stable digest over every tensor reachable in the checkpoint."""
    h = hashlib.sha256()

    def walk(o, path=""):
        if torch.is_tensor(o):
            h.update(path.encode())
            h.update(o.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(o, dict):
            for k in sorted(o, key=str):
                walk(o[k], f"{path}/{k}")
        elif isinstance(o, (list, tuple)):
            for i, v in enumerate(o):
                walk(v, f"{path}[{i}]")
        elif hasattr(o, "state_dict"):
            walk(o.state_dict(), path + "/<sd>")

    walk(obj)
    return h.hexdigest()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def leaks(value):
    if not isinstance(value, str):
        return False
    low = value.replace("/", "\\").lower()
    return ("\\" in low and ":" in low[:3]) or "research_ws" in low \
        or "yolo26-nms-free" in low


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    if not os.path.isdir(WDIR):
        raise SystemExit(f"ERROR: no weights_release/ under {ROOT}")

    sums = []
    for fn in sorted(os.listdir(WDIR)):
        if not fn.endswith(".pt"):
            continue
        p = os.path.join(WDIR, fn)
        ck = torch.load(p, map_location="cpu", weights_only=False)
        ta = ck.get("train_args") or {}
        found = {k: v for k, v in ta.items() if k in PATH_KEYS and leaks(v)}
        extra = {k: v for k, v in ta.items() if k not in PATH_KEYS and leaks(v)}

        print(f"\n{fn}")
        for k, v in {**found, **extra}.items():
            print(f"  train_args[{k}] = {v!r}")
        if not found and not extra:
            print("  no local paths in train_args")

        if args.check:
            sums.append((fn, sha256_file(p)))
            continue

        before = tensor_digest(ck)
        # Ultralytics keeps more than one copy of the training arguments: the
        # top-level `train_args` dict, and another on the model (and EMA)
        # object as `.args`. Sanitizing only the first leaves the paths intact
        # inside the pickled module, where a plain `grep` over the .pt still
        # finds them. Walk everything reachable instead.
        stem = os.path.splitext(fn)[0]

        def scrub(d):
            n = 0
            for k, v in list(d.items()):
                if not leaks(v):
                    continue
                if k in ("model", "weights"):
                    d[k] = BASE.get(fn, os.path.basename(str(v).replace("\\", "/")))
                elif k == "data":
                    d[k] = DATA_CFG.get(fn, os.path.basename(str(v).replace("\\", "/")))
                elif k == "project":
                    d[k] = "runs"
                elif k == "save_dir":
                    d[k] = "runs/" + stem
                elif k == "resume":
                    d[k] = True
                else:
                    d[k] = os.path.basename(str(v).replace("\\", "/"))
                n += 1
            return n

        scrubbed = 0
        seen = set()

        def walk(o, depth=0):
            nonlocal scrubbed
            if depth > 8 or id(o) in seen:
                return
            seen.add(id(o))
            if isinstance(o, dict):
                scrubbed += scrub(o)
                for v in o.values():
                    walk(v, depth + 1)
            elif isinstance(o, (list, tuple)):
                for v in o:
                    walk(v, depth + 1)
            else:
                for attr in ("args", "overrides", "yaml"):
                    if hasattr(o, attr):
                        try:
                            walk(getattr(o, attr), depth + 1)
                        except Exception:
                            pass

        walk(ck)
        print(f"  scrubbed {scrubbed} path field(s) across all copies")

        buf = io.BytesIO()
        torch.save(ck, buf)
        buf.seek(0)
        after = tensor_digest(torch.load(buf, map_location="cpu",
                                         weights_only=False))
        if before != after:
            raise SystemExit(f"ERROR: {fn} tensors changed; refusing to write")
        with open(p, "wb") as f:
            f.write(buf.getvalue())
        print(f"  rewritten; tensor digest unchanged ({before[:16]}...)")
        sums.append((fn, sha256_file(p)))

    if args.apply:
        with open(os.path.join(WDIR, "SHA256SUMS"), "w", encoding="utf-8",
                  newline="\n") as f:
            for fn, s in sums:
                f.write(f"{s}  {fn}\n")
        print(f"\nwrote {os.path.join(WDIR, 'SHA256SUMS')}")


if __name__ == "__main__":
    main()
