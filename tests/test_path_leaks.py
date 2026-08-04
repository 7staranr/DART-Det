# -*- coding: utf-8 -*-
"""leaks() must flag machine paths without flagging repository-relative ones.

Both directions have failed here. It first missed nested and POSIX paths, so
the released checkpoints kept the training workstation's directory layout; then
the POSIX test was written as an unanchored substring, and './data/splits/x.txt'
matched '/data/' -- which would have rewritten a legitimate relative field to a
bare basename. The false-positive direction is the quieter one: nothing errors,
a path field just silently loses its prefix.

    python tests/test_path_leaks.py
"""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
sys.stdout.reconfigure(encoding="utf-8")

import sanitize_checkpoints as sc  # noqa: E402

B = chr(92)
CASES = [
    # must NOT be flagged: these are legitimate values inside the checkpoint
    ("./data/splits/example.txt",          False, "repo-relative"),
    ("data/splits/example.txt",            False, "repo-relative"),
    ("configs/visdrone.yaml",              False, "repo-relative"),
    ("runs/ft_visdrone_yolo26n_1280",      False, "repo-relative"),
    ("yolo26n.pt",                         False, "bare filename"),
    ("",                                   False, "empty"),
    # must be flagged: these identify a machine
    ("/home/alice/project/x.pt",           True,  "POSIX home"),
    ("/mnt/scratch/run/y.yaml",            True,  "POSIX scratch"),
    ("/tmp/build/z",                       True,  "POSIX tmp"),
    ("C:" + B + "Users" + B + "bob" + B + "z.pt", True, "Windows drive"),
    ("E:/Programming/research_ws/x",       True,  "drive, forward slashes"),
    (B * 2 + "server" + B + "share" + B + "a.pt", True, "UNC share"),
]


def main():
    print("path-leak classifier")
    bad = 0
    for value, want, why in CASES:
        got = sc.leaks(value)
        ok = got == want
        bad += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  leaks({value!r}) = {got}"
              + ("" if ok else f"  -- expected {want} ({why})"))
    print("\nall cases hold" if not bad else f"\n{bad} failure(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
