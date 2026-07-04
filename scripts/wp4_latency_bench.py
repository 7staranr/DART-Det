"""GPU-time latency benchmark (edge latency and gate latency axis).

The per-image predict() total is dominated by Python/IO overhead; it is NOT a
fair measure of the decode cost of relaxing max_det. This times the pure model
forward on a fixed batched tensor with CUDA events + synchronize, comparing
max_det=300 vs 1000 (and an o2m+NMS reference). Reports ms/img, % delta, and
the decode fraction (head topk) of total inference. Decides whether a budget
controller has any real latency-Pareto to exploit.

Usage: python wp4_latency_bench.py --weights <pt> --imgsz N --device D
  [--batch 8] [--iters 100]
"""
import argparse
import time

import numpy as np
import torch


def bench(model, x, max_det, iters, trials=8):
    """Return (mean ms/img, std ms/img) over `trials` independent timing runs
    of `iters` forward passes each, to expose run-to-run noise."""
    model.model[-1].max_det = max_det
    per_trial = []
    with torch.no_grad():
        for _ in range(20):                      # warmup
            model(x)
        torch.cuda.synchronize()
        for _ in range(trials):
            t0 = time.time()
            for _ in range(iters):
                model(x)
            torch.cuda.synchronize()
            per_trial.append((time.time() - t0) / iters / x.shape[0] * 1000)
    a = np.array(per_trial)
    return a.mean(), a.std()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    from ultralytics import YOLO

    dev = f"cuda:{args.device}"
    y = YOLO(args.weights)
    m = y.model.to(dev).eval()
    x = torch.rand(args.batch, 3, args.imgsz, args.imgsz, device=dev)

    print(f"batch={args.batch} imgsz={args.imgsz} iters={args.iters} "
          f"trials=8 weights={args.weights.split(chr(92))[-1]}")
    print(f"{'config':>16} {'ms/img':>10} {'std':>7}")
    lat, sd = {}, {}
    for k in (300, 600, 1000, 2000):
        lat[k], sd[k] = bench(m, x, k, args.iters)
        print(f"{'e2e max_det='+str(k):>16} {lat[k]:>10.3f} {sd[k]:>7.3f}",
              flush=True)
    base = lat[300]
    pooled_sd = np.sqrt(np.mean([v ** 2 for v in sd.values()]))
    print(f"\ndecode-cost delta (vs max_det=300; pooled std {pooled_sd:.3f} ms):")
    for k in (600, 1000, 2000):
        d_ms = lat[k] - base
        d_pct = d_ms / base * 100
        sig = "WITHIN NOISE" if abs(d_ms) < 2 * pooled_sd else "outside noise"
        print(f"  300->{k}: {d_ms:+.3f} ms/img ({d_pct:+.1f}%)  [{sig}]")
    print(f"\nInterpretation: total fwd ~{base:.2f} ms/img; deltas vs the pooled "
          f"std ({pooled_sd:.3f} ms) decide whether budget affects latency.")


if __name__ == "__main__":
    main()
