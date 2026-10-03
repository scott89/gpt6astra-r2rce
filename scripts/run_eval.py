#!/usr/bin/env python3
"""Zero-shot R2R-CE evaluation entry point.

Examples
  # validate harness + metrics offline, no network
  python scripts/run_eval.py --policy scripted --episodes 3 --tag scripted-smoke

  # the actual reproduction: an API VLM standing in for the paper's frontier model
  python scripts/run_eval.py --policy vlm --model glm-5.3-flash --episodes 3 --tag glm-smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", choices=["vlm", "scripted"], default="vlm")
    ap.add_argument("--subset-file", default="subset/r2rce100.json")
    ap.add_argument("--task-yaml", default="configs/task_r2rce_512.yaml")
    ap.add_argument(
        "--ext-lib",
        default="/root/autodl-tmp/lavira-code-main",
        help="Repo providing habitat_extensions (dataset/measure/sensor registration).",
    )
    ap.add_argument("--tag", default=None, help="Run name; defaults to <policy>-<timestamp>.")
    ap.add_argument("--episodes", type=int, default=0, help="Cap on episodes to run this invocation (0 = all pending).")
    ap.add_argument("--order", choices=["round-robin", "file"], default="round-robin",
                    help="round-robin spreads the first N episodes across scenes instead of taking file order.")
    ap.add_argument("--max-actions", type=int, default=500, help="Paper budget: 500 primitives per episode.")
    ap.add_argument("--time-limit", type=float, default=2400.0, help="Paper budget: 2400 s wall clock per episode.")
    ap.add_argument("--resume", action="store_true", help="Skip episodes already present in the run dir.")
    ap.add_argument("--only", default=None, metavar="IDS",
                    help="Comma-separated episode ids to restrict this invocation to.")
    ap.add_argument("--shard", default=None, metavar="I/N",
                    help="Run every Nth pending episode, offset I (e.g. 0/5, 1/5). Lets one run fan out.")
    ap.add_argument("--report-only", action="store_true",
                    help="Recompute stats.json from the episodes dir and exit; used to merge parallel shards.")
    # VLM backend: any OpenAI-compatible chat.completions endpoint
    ap.add_argument("--model", default="glm-5.3-flash")
    ap.add_argument("--base-url", default="https://ark.cn-beijing.volces.com/api/coding/v3")
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--reasoning-effort", choices=["minimal", "low", "medium", "high", "none"], default="minimal",
                    help="Thinking budget on endpoints that expose it; 'minimal' = the paper's non-reasoning agent.")
    ap.add_argument("--thinking", action="store_true",
                    help="Shorthand for --reasoning-effort high (the paper's 'ultra reasoning').")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--history-frames", type=int, default=4, help="Camera frames sent per decision, oldest first.")
    ap.add_argument("--max-program", type=int, default=8, help="Longest primitive program accepted per decision.")
    ap.add_argument("--max-consecutive-forward", type=int, default=4,
                    help="Guardrail: cut a program after this many blind MOVE_FORWARD (0 disables).")
    ap.add_argument("--no-stop-verification", action="store_true",
                    help="Disable the look-before-you-leap tilt that precedes the first STOP.")
    # misc
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--save-images", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(PROJ)
    sys.path.insert(0, PROJ)

    from zsvln import runner
    from zsvln.policy import VLMPolicy, ScriptedPolicy

    runner.set_headless(args.gpu)

    with open(args.subset_file) as f:
        subset = json.load(f)
    all_ids = [int(i) for i in subset["episode_ids"]]
    scene_of = {int(e["episode_id"]): e["scene"] for e in subset.get("episodes", [])}

    if args.order == "round-robin":
        from itertools import zip_longest

        by_scene: dict = {}
        for ep_id in all_ids:
            by_scene.setdefault(scene_of.get(ep_id, "?"), []).append(ep_id)
        rings = zip_longest(*[ids for _, ids in sorted(by_scene.items())])
        all_ids = [i for ring in rings for i in ring if i is not None]

    tag = args.tag or f"{args.policy}-{time.strftime('%m%d-%H%M%S')}"
    run_dir = os.path.join("runs", tag)
    ep_dir = os.path.join(run_dir, "episodes")
    os.makedirs(ep_dir, exist_ok=True)

    if args.report_only:
        records = [
            json.load(open(os.path.join(ep_dir, fn)))
            for fn in sorted(os.listdir(ep_dir))
            if fn.endswith(".json")
        ]
        if not records:
            print(f"{ep_dir} is empty.")
            return
        merged = runner.aggregate(records)
        merged["tag"] = tag
        merged["episodes_run"] = [r["episode_id"] for r in records]
        runner.write_json(os.path.join(run_dir, "stats.json"), {"this_run": merged, "all_episodes": merged})
        print(json.dumps(merged, indent=2, default=str))
        print(f"merged {len(records)} episode records into {run_dir}/stats.json")
        return

    done_ids = set()
    if args.resume:
        done_ids = {int(fn[:-5]) for fn in os.listdir(ep_dir) if fn.endswith(".json")}
    pending = [i for i in all_ids if i not in done_ids]
    if args.only:
        wanted = {int(x) for x in args.only.replace(" ", "").split(",")}
        pending = [i for i in pending if i in wanted]
    if args.episodes > 0:
        pending = pending[: args.episodes]
    shard_i, shard_n = 0, 1
    if args.shard:
        shard_i, shard_n = (int(x) for x in args.shard.split("/"))
        if not 0 <= shard_i < shard_n:
            raise SystemExit(f"--shard must be I/N with 0 <= I < N, got {args.shard}")
        pending = pending[shard_i::shard_n]
    if not pending:
        print("Nothing pending. Remove --resume or pick a new --tag.")
        return

    config = runner.load_task_config(args.ext_lib, args.task_yaml, pending, opts=["SEED", args.seed])
    env, budget, action_index = runner.build_env(config, args.max_actions)
    print(
        f"run={tag} policy={args.policy} pending={len(pending)} budget={budget} actions/ep "
        f"time_limit={args.time_limit}s rgb={config.SIMULATOR.RGB_SENSOR.WIDTH}x{config.SIMULATOR.RGB_SENSOR.HEIGHT}",
        flush=True,
    )
    covered = sorted({scene_of.get(i, "?") for i in pending})
    print(f"scenes covered: {len(covered)}/{len(set(scene_of.values()))} -> {', '.join(covered)}", flush=True)

    if args.policy == "vlm":
        from zsvln.backend import ChatBackend

        effort = None if args.reasoning_effort == "none" else ("high" if args.thinking else args.reasoning_effort)
        backend = ChatBackend(
            model=args.model,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
            reasoning_effort=effort,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        policy = VLMPolicy(
            backend,
            history_frames=args.history_frames,
            max_program=args.max_program,
            max_consecutive_forward=args.max_consecutive_forward,
            verify_stop=not args.no_stop_verification,
        )
    else:
        policy = ScriptedPolicy(seed=args.seed, stop_after=min(60, budget))

    records = []
    for idx, ep_id in enumerate(pending, 1):
        frame_sink = None
        if args.save_images:
            frame_dir = os.path.join(run_dir, "frames", str(ep_id))
            os.makedirs(frame_dir, exist_ok=True)

            def frame_sink(step_no, rgb, _d=frame_dir):
                from PIL import Image

                Image.fromarray(rgb.astype("uint8")).save(os.path.join(_d, f"step{step_no:04d}.png"))

        t0 = time.time()
        if hasattr(policy, "backend"):
            policy.backend.reset_stats()  # per-episode cost snapshot, not shard-cumulative
        try:
            record = runner.run_episode(env, policy, budget, args.time_limit, action_index, frame_sink)
        except Exception as error:  # noqa: BLE001 - keep the sweep alive, log the cause
            print(f"[{idx}/{len(pending)}] ep {ep_id} ERROR {type(error).__name__}: {error}", flush=True)
            continue
        if hasattr(policy, "backend"):
            record["backend_stats"] = dict(policy.backend.stats)
            record["stop_deferrals"] = getattr(policy, "stop_deferrals", 0)
        path = os.path.join(ep_dir, f"{record['episode_id']}.json")
        runner.write_json(path, record)
        records.append(record)
        m = record["metrics"]
        print(
            f"[{idx}/{len(pending)}] ep={record['episode_id']} scene={record['scene_id']} "
            f"acts={record['actions_taken']} calls={record['policy_calls']} term={record['termination']} "
            f"dist={m.get('distance_to_goal', float('nan')):.2f} "
            f"success={int(m.get('success', False))} os={int(m.get('oracle_success', False))} "
            f"spl={m.get('spl', 0.0):.3f} ({time.time() - t0:.0f}s)",
            flush=True,
        )

    summary = runner.aggregate(records)
    summary["tag"] = tag
    summary["policy"] = args.policy
    summary["model"] = args.model if args.policy == "vlm" else None
    summary["reasoning_effort"] = effort if args.policy == "vlm" else None
    summary["guardrails"] = {
        "max_consecutive_forward": args.max_consecutive_forward,
        "stop_verification": not args.no_stop_verification,
        "history_frames": args.history_frames,
        "max_program": args.max_program,
    }
    summary["max_actions"] = budget
    summary["time_limit_s"] = args.time_limit
    summary["subset_file"] = args.subset_file
    summary["episodes_run"] = [r["episode_id"] for r in records]
    if hasattr(policy, "backend"):
        summary["backend"] = dict(policy.backend.stats)  # last episode only; totals are in the aggregate keys
    summary["cumulative"] = runner.aggregate(
        [json.load(open(os.path.join(ep_dir, fn))) for fn in sorted(os.listdir(ep_dir)) if fn.endswith(".json")]
    )
    runner.write_json(os.path.join(run_dir, "stats.json" if shard_n == 1 else f"stats-shard{shard_i}.json"),
                      {"this_run": summary, "all_episodes": summary["cumulative"]})

    print("\n=== this run ===")
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {len(records)} episode records to {run_dir}/")


if __name__ == "__main__":
    main()
