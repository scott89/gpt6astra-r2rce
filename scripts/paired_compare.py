#!/usr/bin/env python3
"""Paired comparison of two runs over the same episodes.

Usage: python scripts/paired_compare.py runs/glm-full runs/glm-med100

Only episodes present in BOTH directories are compared, so the two configurations are
measured on identical tasks. Success is a paired binary outcome (exact sign test on the
discordant pairs); everything continuous gets a paired bootstrap CI on the per-episode delta.
"""
from __future__ import annotations

import json
import os
import random
import sys
from math import comb

KEYS = [("success", "{:.1%}"), ("osr", "{:.1%}"), ("ne", "{:.2f}"), ("spl", "{:.1%}"),
        ("actions", "{:.0f}"), ("calls", "{:.1f}"), ("prompt_tokens", "{:.0f}"), ("api_s", "{:.0f}")]


def load(directory):
    out = {}
    root = os.path.join(directory, "episodes")
    for fn in sorted(os.listdir(root)):
        if not fn.endswith(".json"):
            continue
        r = json.load(open(os.path.join(root, fn)))
        m = r["metrics"]
        out[int(fn[:-5])] = {
            "scene": r["scene_id"],
            "success": float(bool(m.get("success"))),
            "osr": float(bool(m.get("oracle_success"))),
            "ne": float(m.get("distance_to_goal") or 0.0),
            "spl": float(m.get("spl") or 0.0),
            "actions": int(r.get("actions_taken") or 0),
            "calls": int(r.get("policy_calls") or 0),
            "prompt_tokens": int(r.get("prompt_tokens") or 0),
            "api_s": float(r.get("api_latency_s") or 0.0),
        }
    return out


def mean(v):
    return sum(v) / len(v)


def paired_boot(da, n=20000, seed=0):
    random.seed(seed)
    means = sorted(mean([da[random.randrange(len(da))] for _ in da]) for _ in range(n))
    return mean(da), means[int(0.025 * n)], means[int(0.975 * n)]


def sign_test(plus, minus):
    n = plus + minus
    if n == 0:
        return 1.0
    k = min(plus, minus)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    a, b = load(sys.argv[1]), load(sys.argv[2])
    common = sorted(set(a) & set(b))
    print(f"A={sys.argv[1]} ({len(a)} eps)  B={sys.argv[2]} ({len(b)} eps)  paired on {len(common)} eps")
    if not common:
        raise SystemExit("no shared episodes")

    print(f"\n{'metric':<16}{'A':>9}{'B':>9}{'B-A':>10}   95% CI of paired delta")
    for key, fmt in KEYS:
        da = [b[k][key] - a[k][key] for k in common]
        m, lo, hi = paired_boot(da)
        note = ""
        if key == "success":
            plus = sum(1 for k in common if b[k]["success"] > a[k]["success"])
            minus = sum(1 for k in common if b[k]["success"] < a[k]["success"])
            note = f"  [B better {plus}, A better {minus}, exact p={sign_test(plus, minus):.3f}]"
        print(f"{key:<16}{fmt.format(mean([a[k][key] for k in common])):>9}"
              f"{fmt.format(mean([b[k][key] for k in common])):>9}{m:>+10.3f}   [{lo:+.3f}, {hi:+.3f}]{note}")

    print("\nnewly successful (A miss -> B hit):", [k for k in common if b[k]["success"] > a[k]["success"]])
    print("newly failing    (A hit -> B miss):", [k for k in common if b[k]["success"] < a[k]["success"]])
    for label, cut in (("NE in 3-5m", lambda r: 3 <= r["ne"] < 5), (">300 actions", lambda r: r["actions"] > 300)):
        print(f"{label}: A={sum(1 for k in common if cut(a[k]))} B={sum(1 for k in common if cut(b[k]))}")

    by_scene = {}
    for k in common:
        by_scene.setdefault(a[k]["scene"], []).append((a[k]["success"], b[k]["success"], b[k]["ne"] - a[k]["ne"]))
    print(f"\n{'scene':<14}{'n':>3}{'SR_A':>7}{'SR_B':>7}{'dNE':>8}")
    for s, v in sorted(by_scene.items(), key=lambda x: -mean([i[1] for i in x[1]])):
        print(f"{s:<14}{len(v):>3}{mean([i[0] for i in v]):>7.0%}{mean([i[1] for i in v]):>7.0%}"
              f"{mean([i[2] for i in v]):>+8.2f}")


if __name__ == "__main__":
    main()
