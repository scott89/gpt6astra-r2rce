"""Zero-shot VLN-CE rollout: budgets, action execution, metrics, reporting."""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np


def set_headless(gpu_id: int = 0) -> None:
    os.environ.setdefault("EGL_PLATFORM", "surfaceless")
    os.environ.setdefault("MAGNUM_LOG", "quiet")
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.pop("DISPLAY", None)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


def load_task_config(habitat_ext_root: str, task_yaml: str, episode_ids: List[int], opts: List[str]):
    root = os.path.abspath(habitat_ext_root)
    if root not in os.sys.path:
        os.sys.path.insert(0, root)
    from habitat_extensions.config.default import get_extended_config

    merge = list(opts) + ["DATASET.EPISODES_ALLOWED", list(episode_ids)]
    config = get_extended_config(config_paths=[task_yaml], opts=merge)
    return config


def build_env(config, max_actions: int):
    import habitat

    env = habitat.Env(config=config)
    budget = min(int(config.ENVIRONMENT.MAX_EPISODE_STEPS), max_actions)
    # POSSIBLE_ACTIONS order is what habitat uses to build the task action space,
    # and it matches habitat-lab's canonical HabitatSimActions indices.
    action_index = {name: i for i, name in enumerate(config.TASK.POSSIBLE_ACTIONS)}
    return env, budget, action_index


def run_episode(env, policy, max_actions: int, time_limit: float,
                action_index: Dict[str, int], frame_sink: Optional[Any] = None) -> Dict[str, Any]:
    obs = env.reset()
    episode = env.current_episode
    instruction_obj = episode.instruction
    instruction = getattr(instruction_obj, "text", None) or getattr(instruction_obj, "instruction_text", "")
    policy.reset()
    started = time.time()
    used = 0
    calls = 0
    api_latency = 0.0
    prompt_tokens = 0
    completion_tokens = 0
    steps: List[Dict[str, Any]] = []
    termination = "unknown"
    done = False
    last_parse_failure_at = -1

    while used < max_actions and not done:
        if time.time() - started > time_limit:
            termination = "timeout"
            break
        decision = policy.decide(instruction, obs["rgb"], used, max_actions)
        calls += 1
        if not decision["parsed_ok"]:
            last_parse_failure_at = len(steps)
        thought = decision["thought"]
        api_latency += float(decision.get("latency_s", 0.0))
        prompt_tokens += int(decision.get("prompt_tokens", 0))
        completion_tokens += int(decision.get("completion_tokens", 0))
        executed: List[str] = []
        for name in [a for a in decision["actions"] if a in action_index]:
            if used >= max_actions or env.episode_over:
                break
            if time.time() - started > time_limit:
                break
            obs = env.step({"action": action_index[name]})
            used += 1
            executed.append(name)
            if frame_sink is not None:
                frame_sink(used, obs["rgb"])
            if name == "STOP":
                termination = "stop"
                break
            if env.episode_over:
                break
        policy.note(obs["rgb"], executed)
        steps.append({"n": used, "actions": executed, "thought": thought,
                      "latency_s": decision.get("latency_s", 0.0),
                      "prompt_tokens": decision.get("prompt_tokens", 0)})
        done = env.episode_over
        if termination == "stop" or done:
            break
    if termination == "unknown":
        termination = "budget_exhausted" if used >= max_actions else "env_done"

    metrics = {k: v for k, v in env.get_metrics().items() if k != "position"}
    pos = env.get_metrics().get("position") or {}
    distances = list(pos.get("distance", [])) if isinstance(pos, dict) else []
    elapsed = time.time() - started
    stop_called = bool(getattr(env.task, "is_stop_called", False))

    success = bool(metrics.get("success", False))
    return {
        "episode_id": int(episode.episode_id),
        "scene_id": episode.scene_id.split("/")[-2],
        "instruction": instruction.strip(),
        "actions_taken": used,
        "policy_calls": calls,
        "api_latency_s": round(api_latency, 2),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "termination": termination,
        "stop_called": stop_called,
        "elapsed_s": round(elapsed, 2),
        "steps": steps,
        "last_parse_failure_at": last_parse_failure_at,
        "distance_trace": [round(float(d), 3) for d in distances],
        "final_distance": round(float(distances[-1]), 3) if distances else None,
        "min_distance": round(float(min(distances)), 3) if distances else None,
        "metrics": {k: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v)) for k, v in metrics.items()},
        "success": success,
    }


METRIC_KEYS = ("success", "oracle_success", "distance_to_goal", "spl", "path_length", "steps_taken")


def aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(records)
    if n == 0:
        return {"num_episodes": 0}
    out: Dict[str, Any] = {"num_episodes": n}
    for key in METRIC_KEYS:
        vals = [r["metrics"].get(key) for r in records if r["metrics"].get(key) is not None]
        if vals:
            out[key] = round(float(np.mean(vals)), 4)
    out["termination"] = {
        t: sum(1 for r in records if r["termination"] == t)
        for t in sorted({r["termination"] for r in records})
    }
    out["mean_actions_per_episode"] = round(float(np.mean([r["actions_taken"] for r in records])), 2)
    out["mean_policy_calls_per_episode"] = round(float(np.mean([r["policy_calls"] for r in records])), 2)
    out["episodes_with_unparsable_reply"] = sum(1 for r in records if r["last_parse_failure_at"] >= 0)
    out["mean_stop_deferrals"] = round(float(np.mean([r.get("stop_deferrals", 0) for r in records])), 2)
    out["api_latency_s_total"] = round(sum(r.get("api_latency_s", 0.0) for r in records), 1)
    out["prompt_tokens_total"] = sum(r.get("prompt_tokens", 0) for r in records)
    out["completion_tokens_total"] = sum(r.get("completion_tokens", 0) for r in records)
    return out


def write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Atomic replace: parallel shards read each other's episode files to build cumulative stats.
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp, path)
