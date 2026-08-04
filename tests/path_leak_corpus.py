# -*- coding: utf-8 -*-
"""The single corpus both path gates are checked against.

There are two independent path checks in this repository: leaks() inside
scripts/sanitize_checkpoints.py, which decides what to rewrite inside a
checkpoint, and the grep pattern in .github/workflows/smoke.yml, which decides
whether tracked text carries a workstation path. A comment used to claim they
shared a pattern set "so the two gates cannot drift apart". They did not share
anything, and they had already drifted: the scanner accepted E:/work/... while
the regex's drive branch only matched a backslash.

They still cannot share an implementation -- one is Python, the other is an ERE
inside a YAML workflow. What they can share is this corpus, which both are
tested against. Drift now fails a test instead of going unnoticed.

Every case carries a reason, because a case whose expected value can be reached
by the wrong rule is worse than no case at all: the first version of this table
used "E:/Programming/research_ws/x" to test forward-slash drive handling, and
that string also contains the codename, so it passed through the fallback even
when the drive rule missed it entirely.
"""

B = chr(92)

# (value, should_be_flagged, why)
CASES = [
    # --- must NOT be flagged: legitimate values inside a checkpoint or a file
    ("./data/splits/example.txt", False, "repo-relative; the '/data/' regression"),
    ("data/splits/example.txt",   False, "repo-relative"),
    ("configs/visdrone.yaml",     False, "repo-relative"),
    ("runs/ft_visdrone_yolo26n_1280", False, "repo-relative"),
    ("results/per_image/wp1_ft/x.csv", False, "repo-relative"),
    ("yolo26n.pt",                False, "bare filename"),
    ("",                          False, "empty"),
    # URLs contain '//' and must not be mistaken for a UNC share; the README
    # and the workflow both carry real ones.
    ("https://github.com/7staranr/DART-Det", False, "https URL"),
    ("https://download.pytorch.org/whl/cpu", False, "https URL"),

    # --- must be flagged: these identify a machine. None of them contains the
    #     codename or 'research_ws', so each exercises exactly one rule.
    ("/home/alice/project/x.pt",  True,  "POSIX home"),
    ("/mnt/scratch/run/y.yaml",   True,  "POSIX scratch"),
    ("/tmp/build/z",              True,  "POSIX tmp"),
    ("/root/train/out.pt",        True,  "POSIX root"),
    ("//home/alice/x.pt",         True,  "doubled leading slash"),
    ("C:" + B + "Users" + B + "bob" + B + "z.pt", True, "Windows drive, backslash"),
    ("E:/work/project/x.pt",      True,  "Windows drive, forward slash"),
    (B * 2 + "server" + B + "share" + B + "a.pt", True, "UNC, backslash"),
    ("//server/share/a.pt",       True,  "UNC, forward slash"),

    # --- the pre-release codename, on its own
    ("YOLO26-NMS-FREE",           True,  "pre-release codename, upper case"),
    ("some/path/research_ws/x",   True,  "workspace name"),
]
