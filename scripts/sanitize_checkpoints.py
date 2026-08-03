# -*- coding: utf-8 -*-
"""Normalize the released checkpoints' path metadata.

Ultralytics records the training invocation inside the checkpoint, so a
finetune produced on a workstation carries that machine's absolute paths in
`train_args` (model, data, project, save_dir, name, source, weights, and
resume -- the set in PATH_KEYS plus resume). Those fields are
useless to anyone else -- they point at directories that do not exist on their
machine -- and they publish the author's local layout.

This rewrites only those metadata strings. The check it performs is exactly
this: every tensor reachable in the checkpoint is digested before and after,
and the write is refused if the digests differ. That proves no weight or EMA
buffer changed. It does not by itself prove that non-tensor metadata other
than the path fields is untouched -- for that, rely on scrub() only ever
assigning to keys whose current value satisfies leaks().

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
    """Digest over the tensors reachable through containers and state_dict().

    Scope, stated exactly: this covers tensors found in dicts, lists/tuples,
    and any object exposing state_dict() -- which is what holds the weights and
    the EMA buffers, and therefore what backs the claim that neither changed.
    It is not a digest of literally every tensor in the pickle.

    Deliberately identity-blind: an earlier revision added a vars() walk with an
    id()-based cycle guard, and because object identities and aliasing differ
    between two deserializations the guard pruned different subtrees each time.
    The digest then disagreed with itself on a checkpoint nothing had touched,
    and the script refused to write a file it had correctly left alone. A
    verification function that is not reproducible is worse than a narrower one.
    """
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


# Directory prefixes that identify a machine rather than a repository. Kept
# broad on purpose: a false positive costs one basename, a false negative
# publishes someone's home directory.
_POSIX_ROOTS = ("/home/", "/users/", "/mnt/", "/media/", "/opt/", "/srv/",
                "/var/", "/root/", "/tmp/", "/private/", "/data/")


def leaks(value):
    """Does this string look like a path on the machine that trained the model?"""
    if not isinstance(value, str) or not value.strip():
        return False
    low = value.lower()
    win = low.replace("/", "\\")
    if "\\" in win and len(win) > 2 and win[1] == ":":       # C:\... , E:/...
        return True
    if any(low.startswith(r) or ("/" + r.strip("/") + "/") in low
           for r in _POSIX_ROOTS):                            # /home/alice/...
        return True
    if low.startswith("\\\\"):                                # UNC share
        return True
    return "research_ws" in low or "yolo26-nms-free" in low


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    if not os.path.isdir(WDIR):
        raise SystemExit(f"ERROR: no weights_release/ under {ROOT}")

    sums = []
    staged = []
    any_found = False
    for fn in sorted(os.listdir(WDIR)):
        if not fn.endswith(".pt"):
            continue
        p = os.path.join(WDIR, fn)
        ck = torch.load(p, map_location="cpu", weights_only=False)

        # Traverse dicts, sequences and object attribute dicts to depth 8.
        # Ultralytics keeps a second copy of the arguments on the model (and
        # EMA) object, and an earlier version of this script checked only the
        # top-level train_args, reporting "clean" while `grep` still found
        # paths in the file. The depth bound is a cycle/blow-up guard: strings
        # nested deeper than 8 levels would not be seen, which is why the CI
        # also greps the raw bytes.
        found = []
        seen = set()

        def scan(o, path="", depth=0):
            # Check the node itself rather than only dict values and
            # attributes: a path sitting directly inside a list has no parent
            # key, and the earlier version walked straight past it while
            # reporting that sequences were covered.
            if leaks(o):
                found.append((path, o))
                return
            if depth > 8 or id(o) in seen:
                return
            seen.add(id(o))
            if isinstance(o, dict):
                for k, v in o.items():
                    scan(v, f"{path}[{k!r}]", depth + 1)
            elif isinstance(o, (list, tuple)):
                for i, v in enumerate(o):
                    scan(v, f"{path}[{i}]", depth + 1)
            elif hasattr(o, "__dict__"):
                # Real attribute traversal, not three hand-picked names: a
                # pickled model carries its arguments on itself, and naming
                # only the attributes we happen to know about is what let a
                # nested copy survive the first pass.
                try:
                    attrs = vars(o)
                except TypeError:
                    attrs = {}
                for k, v in attrs.items():
                    scan(v, f"{path}.{k}", depth + 1)

        scan(ck, "ckpt")
        print(f"\n{fn}")
        for where, v in found:
            print(f"  {where} = {v!r}")
        if not found:
            print("  no machine-local paths found (dicts, sequences and "
                  "object attributes to depth 8)")

        if args.check:
            sums.append((fn, sha256_file(p)))
            any_found = any_found or bool(found)
            continue

        before = tensor_digest(ck)
        # Ultralytics keeps more than one copy of the training arguments: the
        # top-level `train_args` dict, and another on the model (and EMA)
        # object as `.args`. Sanitizing only the first leaves the paths intact
        # inside the pickled module, where a plain `grep` over the .pt still
        # finds them. Walk everything reachable instead.
        stem = os.path.splitext(fn)[0]

        unexpected = []

        def scrub(d):
            n = 0
            for k, v in list(d.items()):
                if not leaks(v):
                    continue
                if k not in PATH_KEYS and k != "resume":
                    # Only the declared training-path fields are rewritten. A
                    # path anywhere else is surfaced rather than silently
                    # normalized, so the sanitizer's blast radius stays exactly
                    # as wide as it is documented to be.
                    unexpected.append((k, v))
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
                for i, v in enumerate(o):
                    if leaks(v):
                        # No field name here, so no defensible replacement.
                        unexpected.append((f"<sequence>[{i}]", v))
                    walk(v, depth + 1)
            elif hasattr(o, "__dict__"):
                try:
                    attrs = vars(o)
                except TypeError:
                    attrs = {}
                if attrs:
                    scrubbed += scrub(attrs)
                for v in attrs.values():
                    walk(v, depth + 1)

        walk(ck)
        print(f"  scrubbed {scrubbed} path field(s) across all copies")
        if unexpected:
            for k, v in unexpected:
                print(f"  NOT rewritten (outside PATH_KEYS): {k} = {v!r}")
            raise SystemExit(
                f"ERROR: {len(unexpected)} path-looking value(s) sit outside the "
                f"declared path fields {PATH_KEYS}. Widen PATH_KEYS deliberately "
                f"or handle them by hand; this script will not guess.")

        buf = io.BytesIO()
        torch.save(ck, buf)
        buf.seek(0)
        after = tensor_digest(torch.load(buf, map_location="cpu",
                                         weights_only=False))
        if before != after:
            raise SystemExit(f"ERROR: {fn} tensors changed; refusing to write")
        # Stage rather than write: a failure on a later checkpoint would
        # otherwise leave some files rewritten, none of them checksummed, and
        # no single command to tell which state the directory is in.
        staged.append((p, fn, buf.getvalue()))
        print(f"  verified; staged for write (tensor digest unchanged "
              f"{before[:16]}...)")

    if args.apply:
        # All checkpoints verified; commit them together, then checksum.
        for path, fn, blob in staged:
            with open(path, "wb") as f:
                f.write(blob)
            print(f"  wrote {fn}")
        sums = [(fn, sha256_file(path)) for path, fn, _ in staged]
        with open(os.path.join(WDIR, "SHA256SUMS"), "w", encoding="utf-8",
                  newline="\n") as f:
            for fn, s in sums:
                f.write(f"{s}  {fn}\n")
        print(f"\nwrote {os.path.join(WDIR, 'SHA256SUMS')}")
    elif any_found:
        raise SystemExit(
            "\nERROR: machine-local paths remain in the released checkpoints. "
            "Run --apply.")


if __name__ == "__main__":
    main()
