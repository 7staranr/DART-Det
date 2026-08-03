# -*- coding: utf-8 -*-
"""Regression tests for the prediction-cache integrity guards.

These exist because the failure they cover is invisible: an image that
inference skipped used to leave the denominator rather than score zero, so
every rate came out slightly high and nothing in the output said so. The guards
that fix it are themselves easy to get subtly wrong -- the first version filled
in no scale denominators, which reintroduced the same bias one level down --
so they are pinned here rather than checked by eye.

    python tests/test_cache_integrity.py

No pytest dependency: this runs standalone in CI alongside `compileall`.
"""
import csv
import io
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.stdout.reconfigure(encoding="utf-8")

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


def make_split(tmp, with_b=True, duplicate_a=False):
    """Two VisDrone-style images, 3 GT boxes each; img_b optionally uncached."""
    ann = os.path.join(tmp, "annotations")
    os.makedirs(ann, exist_ok=True)
    for name in ("img_a", "img_b"):
        with io.open(os.path.join(ann, name + ".txt"), "w", encoding="utf-8") as f:
            for i in range(3):
                f.write(f"{10 + i * 50},{10 + i * 50},100,100,1,1,0,0\n")
    preds = os.path.join(tmp, "p.jsonl")
    rec_a = {"image": "img_a", "width": 1000, "height": 1000,
             "boxes": [[10, 10, 110, 110, 0.9, 1]]}
    with io.open(preds, "w", encoding="utf-8") as f:
        f.write(json.dumps(rec_a) + "\n")
        if duplicate_a:
            f.write(json.dumps(rec_a) + "\n")
        if with_b:
            f.write(json.dumps({"image": "img_b", "width": 1000,
                                "height": 1000, "boxes": []}) + "\n")
    return ann, preds


def run_eval(ann, preds, out, extra=()):
    return subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", "wp1_eval.py"),
         "--dataset", "visdrone", "--gt", ann, "--preds", preds,
         "--out-prefix", out, *extra],
        capture_output=True, text=True)


def main():
    print("cache-integrity guards")
    with tempfile.TemporaryDirectory() as tmp:
        # 1. a missing image must stop the run, not quietly shrink the denominator
        ann, preds = make_split(tmp, with_b=False)
        r = run_eval(ann, preds, os.path.join(tmp, "o1", "t"))
        check("missing image fails closed by default", r.returncode != 0)
        check("coverage is reported", "coverage:" in r.stdout + r.stderr)

        # 2. --allow-missing must score it zero-recall with its scale
        #    denominators intact -- the defect the first fix shipped with
        r = run_eval(ann, preds, os.path.join(tmp, "o2", "t"), ["--allow-missing"])
        check("--allow-missing succeeds", r.returncode == 0, r.stderr[-300:])
        rows = {x["image"]: x for x in csv.DictReader(
            io.open(os.path.join(tmp, "o2", "t_per_image.csv"), encoding="utf-8"))}
        check("missing image is present as a row", "img_b" in rows)
        if "img_b" in rows:
            b = rows["img_b"]
            scale_sum = sum(int(b[f"n_gt_{s}"]) for s in ("small", "medium", "large"))
            check("its GT stays in the overall denominator", int(b["n_gt"]) == 3)
            check("its GT stays in the scale denominators", scale_sum == 3,
                  f"got {scale_sum}, expected 3")
            check("it scores zero matches", int(b["matched@300"]) == 0)
        check("missing images are listed",
              os.path.exists(os.path.join(tmp, "o2", "t_missing_images.txt")))

        # 3. an empty prediction list is data, not absence: it must not trip the gate
        ann, preds = make_split(tmp, with_b=True)
        r = run_eval(ann, preds, os.path.join(tmp, "o3", "t"))
        check("a legitimately empty prediction record is not 'missing'",
              r.returncode == 0, r.stderr[-300:])

        # 4. duplicate keys are invisible to a set-based coverage check
        ann, preds = make_split(tmp, with_b=True, duplicate_a=True)
        r = run_eval(ann, preds, os.path.join(tmp, "o4", "t"))
        check("duplicate image keys are rejected",
              r.returncode != 0 and "duplicate" in (r.stdout + r.stderr).lower())

        # 5. an empty output-prefix directory component must not crash
        ann, preds = make_split(tmp, with_b=True)
        r = subprocess.run(
            [sys.executable, os.path.join(ROOT, "scripts", "wp1_eval.py"),
             "--dataset", "visdrone", "--gt", ann, "--preds", preds,
             "--out-prefix", "t"],
            capture_output=True, text=True, cwd=tmp)
        check("bare --out-prefix works", r.returncode == 0, r.stderr[-300:])

    print(f"\n{len(FAILURES)} failure(s)" if FAILURES else "\nall guards hold")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
