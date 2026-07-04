"""o2m+NMS dense AP@0.5 across NMS IoU thresholds (VisDrone dense), to show the
one-to-many recovery is not an NMS-tuning artifact."""
import os
import sys

sys.path.insert(0, os.path.join(ROOT, "scripts"))
import wp1_eval as we
import wp4_ap_bootstrap as ab
import iteration1_tables as it

ROOT = os.environ.get("DART_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
E = os.path.join(ROOT, "experiments", "wp1_ft")
vd = we.load_visdrone_gt(os.path.join(ROOT, "data", "VisDrone2019-DET-val", "annotations"))
e2e = ab.load_preds(os.path.join(E, "preds_visdrone_ft_s.jsonl"))
ids = [k for k, v in vd.items() if len(v["gt"]) >= 150 and k in e2e]
print("VisDrone dense n =", len(ids))
print("e2e cap-300   dense AP@0.5 = %.3f" % it.ap50(e2e, vd, ids, 300, 10))
print("e2e cap-1000  dense AP@0.5 = %.3f" % it.ap50(e2e, vd, ids, 1000, 10))
for name, f in [("o2m NMS@0.5", "preds_ft_s_o2m_nms05_ag.jsonl"),
                ("o2m NMS@0.65", "preds_ft_s_o2m_nms065.jsonl"),
                ("o2m NMS@0.7", "preds_ft_s_o2m_nms07_ag.jsonl")]:
    p = os.path.join(E, f)
    if not os.path.exists(p):
        print(name, "MISSING")
        continue
    print("%-13s dense AP@0.5 = %.3f" % (name, it.ap50(ab.load_preds(p), vd, ids, 1000, 10)))
