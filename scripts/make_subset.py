#!/usr/bin/env python3
"""Freeze a deterministic R2R-CE-100 subset: 10 episodes from each of 10 val_unseen scenes.

The paper evaluates "100 episodes covering one task in a 10-scene validation-unseen subset"
but does not publish the episode list, so we define ours explicitly and store it.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import os


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        default="data/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz",
    )
    ap.add_argument("--out", default="subset/r2rce100.json")
    ap.add_argument("--episodes-per-scene", type=int, default=10)
    ap.add_argument("--num-scenes", type=int, default=10)
    args = ap.parse_args()

    with gzip.open(args.dataset, "rt") as f:
        episodes = json.load(f)["episodes"]

    by_scene = collections.defaultdict(list)
    for ep in episodes:
        scene = ep["scene_id"].split("/")[-2]
        by_scene[scene].append(ep)

    # Drop the rarest scenes, matching the paper's 10-scene envelope.
    scenes = [s for s, _ in sorted(by_scene.items(), key=lambda kv: -len(kv[1]))][: args.num_scenes]

    chosen = []
    for scene in scenes:
        pool = sorted(by_scene[scene], key=lambda ep: int(ep["episode_id"]))
        stride = max(1, len(pool) // args.episodes_per_scene)
        picked = pool[::stride][: args.episodes_per_scene]
        chosen.extend(picked)

    payload = {
        "split": "val_unseen",
        "dataset": args.dataset,
        "episodes_per_scene": args.episodes_per_scene,
        "num_scenes": args.num_scenes,
        "scenes": scenes,
        "dropped_scenes": sorted(set(by_scene) - set(scenes)),
        "episode_ids": [int(ep["episode_id"]) for ep in chosen],
        "episodes": [
            {
                "episode_id": int(ep["episode_id"]),
                "scene": ep["scene_id"].split("/")[-2],
                "instruction": ep["instruction"]["instruction_text"],
                "traj_length": len(ep["reference_path"]),
            }
            for ep in sorted(chosen, key=lambda e: int(e["episode_id"]))
        ],
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"val_unseen: {len(episodes)} episodes / {len(by_scene)} scenes")
    print(f"subset: {len(chosen)} episodes over {len(scenes)} scenes -> {args.out}")
    print("scenes:", ", ".join(f"{s}({len(by_scene[s])})" for s in scenes))
    print("dropped:", payload["dropped_scenes"])


if __name__ == "__main__":
    main()
