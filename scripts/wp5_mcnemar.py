"""Paired McNemar test: WP5 (treatment) vs baseline, per-GT detection@300.

Reads the two per-GT CSVs produced by wp1_local_density.py (run via the
ft suite for each weight). Rows are aligned by order (same val images, same
GT order, deterministic). For each density/crowding stratum, counts discordant
pairs (one model detects the GT@300, the other misses) and reports the
exact-binomial / normal-approx McNemar statistic.

Usage:
  python wp5_mcnemar.py --base <baseline_per_gt.csv> --treat <wp5_per_gt.csv>
                        [--base2 ... --treat2 ...]   # pool a 2nd seed
"""
import argparse
import csv
from math import sqrt

from scipy.stats import binomtest


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


STRATA = {
    "ALL GT": lambda r: True,
    "sparse <50": lambda r: int(r["n_gt"]) < 50,
    "dense >=150": lambda r: int(r["n_gt"]) >= 150,
    "crowded maxIoU>=0.3": lambda r: float(r["max_iou_nbr"]) >= 0.3,
    "dense AND crowded": lambda r: int(r["n_gt"]) >= 150
    and float(r["max_iou_nbr"]) >= 0.3,
    "small AND dense": lambda r: r["size"] == "small"
    and int(r["n_gt"]) >= 150,
}


def collect(pairs):
    """pairs: list of (base_rows, treat_rows). Returns {stratum: (b01, b10)}."""
    out = {k: [0, 0] for k in STRATA}
    for base, treat in pairs:
        assert len(base) == len(treat), "row count mismatch"
        for rb, rv in zip(base, treat):
            assert rb["image"] == rv["image"], "alignment mismatch"
            mb, mv = int(rb["m300"]), int(rv["m300"])
            if mb == mv:
                continue
            for k, sel in STRATA.items():
                if sel(rb):
                    if mv == 1:        # treat wins (base missed, treat hit)
                        out[k][0] += 1
                    else:              # base wins
                        out[k][1] += 1
    return out


def report(title, counts):
    print(f"\n=== {title} ===")
    print(f"{'stratum':>22}  {'treat_win':>9} {'base_win':>8} "
          f"{'net':>6} {'z':>7} {'p(2-sided)':>11}")
    for k, (b01, b10) in counts.items():
        n = b01 + b10
        if n == 0:
            print(f"{k:>22}  {'--':>9}")
            continue
        z = (b01 - b10) / sqrt(n)
        p = binomtest(b01, n, 0.5).pvalue
        print(f"{k:>22}  {b01:>9} {b10:>8} {b01 - b10:>+6} "
              f"{z:>+7.2f} {p:>11.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--treat", required=True)
    ap.add_argument("--base2")
    ap.add_argument("--treat2")
    args = ap.parse_args()

    p1 = (load(args.base), load(args.treat))
    report("seed-0", collect([p1]))
    if args.base2 and args.treat2:
        p2 = (load(args.base2), load(args.treat2))
        report("seed-1", collect([p2]))
        report("pooled (seed-0 + seed-1)", collect([p1, p2]))


if __name__ == "__main__":
    main()
