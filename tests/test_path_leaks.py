# -*- coding: utf-8 -*-
"""Both path gates, checked against one corpus.

Two independent gates decide whether a string looks like a workstation path:
leaks() in scripts/sanitize_checkpoints.py, and the grep pattern in
.github/workflows/smoke.yml. They cannot share an implementation -- one is
Python, the other an ERE inside YAML -- so they share tests/path_leak_corpus.py
instead, which is the only thing that stops them drifting apart. They had
already drifted once: the scanner accepted "E:/work/..." while the regex's
drive branch matched only a backslash.

Both directions matter. Missing a path publishes the training machine's
layout; flagging a repository-relative path is quieter and worse, because the
sanitizer would rewrite a legitimate field to a bare basename and nothing would
error.

The regex is read out of the workflow file and run through real grep. Retyping
it here would test a copy, and typing it through a shell has twice mangled the
backslashes badly enough to produce a confidently wrong conclusion.

    python tests/test_path_leaks.py
"""
import io
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")

from path_leak_corpus import CASES          # noqa: E402
import sanitize_checkpoints as sc           # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}"
          + (f"  -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


def test_scanner():
    print("leaks() in sanitize_checkpoints.py")
    for value, want, why in CASES:
        got = sc.leaks(value)
        check(f"leaks({value!r}) == {want}", got == want, f"got {got} ({why})")


def workflow_pattern():
    """The ERE the CI text gate actually runs, read from the workflow."""
    wf = os.path.join(ROOT, ".github", "workflows", "smoke.yml")
    for line in io.open(wf, encoding="utf-8"):
        if "git grep" in line and "-nEIi" in line:
            return line.split("'")[1]
    return None


def test_ci_regex():
    pat = workflow_pattern()
    print("\nCI text-gate regex (read from smoke.yml, run through grep)")
    if pat is None:
        check("workflow pattern found", False, "no git grep -nEIi line")
        return
    d = tempfile.mkdtemp()
    pf, tf = os.path.join(d, "pat"), os.path.join(d, "probe")
    io.open(pf, "w", encoding="utf-8", newline="\n").write(pat + "\n")
    # One case per line, so a line number maps back to a case.
    io.open(tf, "w", encoding="utf-8", newline="\n").write(
        "".join(v.replace("\n", " ") + "\n" for v, _, _ in CASES))
    r = subprocess.run(["grep", "-nEi", "-f", pf, tf],
                       capture_output=True, text=True)
    hit = {int(l.split(":", 1)[0]) for l in r.stdout.splitlines() if ":" in l}
    for i, (value, want, why) in enumerate(CASES, 1):
        if value == "":
            continue                      # an empty line cannot be probed
        got = i in hit
        check(f"regex flags {value!r} == {want}", got == want,
              f"got {got} ({why})")


def main():
    test_scanner()
    test_ci_regex()
    print(f"\n{len(FAILURES)} failure(s)" if FAILURES else "\nboth gates agree with the corpus")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
