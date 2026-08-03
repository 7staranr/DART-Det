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

        # 6. the reported bucket table, not just the per-image rows, must keep
        #    the missing image's ground truth. Checking per-image alone would
        #    miss an aggregate step that filtered the appended rows back out.
        ann, preds = make_split(tmp, with_b=False)
        r = run_eval(ann, preds, os.path.join(tmp, "o6", "t"), ["--allow-missing"])
        buckets = list(csv.DictReader(
            io.open(os.path.join(tmp, "o6", "t_buckets.csv"), encoding="utf-8")))
        total_gt = sum(int(x["total_gt"]) for x in buckets)
        n_images = sum(int(x["n_images"]) for x in buckets)
        check("bucket table keeps both images", n_images == 2, f"got {n_images}")
        check("bucket table keeps all 6 GT", total_gt == 6, f"got {total_gt}")
        rk = [x for x in buckets if int(x["n_images"])]
        recall = sum(float(x["R@300"]) * int(x["total_gt"]) for x in rk) / total_gt
        check("pooled R@300 is 1/6, not 1/3",
              abs(recall - 1 / 6) < 1e-3, f"got {recall:.4f}")

        # 7. the DABA policy script must keep a missing image too. This is the
        #    path where an earlier fix appended rows that zip() then dropped,
        #    and no test covered it.
        r = subprocess.run(
            [sys.executable, os.path.join(ROOT, "scripts", "wp4_budget_policy.py"),
             "--dataset", "visdrone", "--gt", ann, "--preds", preds],
            capture_output=True, text=True)
        check("budget policy fails closed on a partial cache", r.returncode != 0)
        r = subprocess.run(
            [sys.executable, os.path.join(ROOT, "scripts", "wp4_budget_policy.py"),
             "--dataset", "visdrone", "--gt", ann, "--preds", preds,
             "--allow-missing"],
            capture_output=True, text=True)
        out = r.stdout + r.stderr
        check("budget policy runs under --allow-missing", r.returncode == 0,
              out[-300:])
        # Both images hold 3 GT. If the missing one is scored as an empty
        # prediction its GT stays in the denominator and recall is 1/6; if it
        # is dropped -- the bug an earlier fix left in place -- recall is 1/3.
        # columns are: policy  bucket  recall  FP/img  slots
        recalls = sorted({line.split()[2] for line in out.splitlines()
                          if "<50" in line and "policy" not in line
                          and len(line.split()) >= 5})
        check("budget policy scores the missing image rather than dropping it",
              recalls == ["0.1667"], f"printed recalls: {recalls}")

    print(f"\n{len(FAILURES)} failure(s)" if FAILURES else "\nall guards hold")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
