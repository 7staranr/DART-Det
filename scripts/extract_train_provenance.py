# -*- coding: utf-8 -*-
"""Extract the auditable evidence behind training_manifest.csv.

`optimizer=auto` means the requested arguments say nothing about what actually
ran: the framework resolves the optimizer, lr0 and momentum at startup and
prints them once. That single line is the evidence for three manifest columns,
and it lives in a training log that is far too large (and too noisy) to commit.

This ships the line itself plus the SHA256 of the log it came from, so a reader
holding the logs can verify the manifest, and a reader without them can at
least see the exact text the numbers were read from rather than taking the CSV
on trust.

    python scripts/extract_train_provenance.py --runs <dir with the .log files>

Two of the logs are UTF-16 (PowerShell redirection), which a utf-8 read turns
into null-interleaved text that matches nothing, so the encoding is sniffed.
"""
import argparse
import csv
import hashlib
import io
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.environ.get("DART_ROOT",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ANSI = re.compile(r"\x1b\[[0-9;]*m")
OPT = re.compile(r"optimizer:\s*(\w+)\(lr=([0-9.eE+-]+),\s*momentum=([0-9.]+)")


def read_lines(path):
    with open(path, "rb") as f:
        bom = f.read(2)
    enc = "utf-16" if bom in (bytes([255, 254]), bytes([254, 255])) else "utf-8"
    with io.open(path, encoding=enc, errors="replace") as f:
        return f.readlines()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", required=True,
                    help="directory holding the training .log files")
    ap.add_argument("--manifest",
                    default=os.path.join(ROOT, "training_manifest.csv"))
    ap.add_argument("--out",
                    default=os.path.join(ROOT, "provenance",
                                         "resolved_optimizer.txt"))
    args = ap.parse_args()

    rows = list(csv.DictReader(io.open(args.manifest, encoding="utf-8")))
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    missing, lines = [], []
    lines.append("Resolved optimizer evidence for training_manifest.csv\n")
    lines.append("Each block is the line the framework printed after resolving\n"
                 "optimizer=auto, plus the SHA256 of the log it was read from.\n"
                 "Regenerate with scripts/extract_train_provenance.py.\n")
    for r in rows:
        log = os.path.join(args.runs, r["source_log"])
        lines.append(f"\n[{r['run']}]  seed={r['seed']}  script={r['training_script']}")
        if not os.path.exists(log):
            lines.append(f"  log not available: {r['source_log']}")
            missing.append(r["run"])
            continue
        lines.append(f"  log: {r['source_log']}  sha256={sha256(log)}")
        hit = next((ANSI.sub("", ln).strip() for ln in read_lines(log)
                    if OPT.search(ANSI.sub("", ln))), None)
        lines.append(f"  {hit}" if hit else "  no resolved-optimizer line found")
        if hit:
            m = OPT.search(hit)
            ok = (m.group(1) == r["resolved_optimizer"]
                  and float(m.group(2)) == float(r["resolved_lr0"])
                  and float(m.group(3)) == float(r["momentum"]))
            lines.append(f"  manifest agrees: {'yes' if ok else 'NO -- MISMATCH'}")
            if not ok:
                missing.append(r["run"] + " (mismatch)")

    io.open(args.out, "w", encoding="utf-8", newline="\n").write(
        "\n".join(lines) + "\n")
    print(f"wrote {args.out} for {len(rows)} run(s)")
    if missing:
        print("  unresolved:", ", ".join(missing))
        return 1
    print("  every manifest row matches its log")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
