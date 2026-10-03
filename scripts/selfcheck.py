#!/usr/bin/env python3
"""Offline checks: action parsing, guardrails, prompt assembly. No network."""
from __future__ import annotations

import os
import sys

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJ)
sys.path.insert(0, PROJ)

from zsvln.policy import VLMPolicy, parse_actions  # noqa: E402

fails = 0


def check(label, got, want):
    global fails
    ok = got == want
    if not ok:
        fails += 1
    print(f"[{'ok' if ok else 'FAIL'}] {label}: {got if ok else f'{got!r} != {want!r}'}")


CASES = [
    ('{"thought": "t", "actions": ["TURN_RIGHT", "MOVE_FORWARD"]}', ["TURN_RIGHT", "MOVE_FORWARD"]),
    ('Sure!\n```json\n{"actions": ["MOVE_FORWARD", "MOVE_FORWARD", "STOP"]}\n```',
     ["MOVE_FORWARD", "MOVE_FORWARD", "STOP"]),
    ('{"actions": ["MOVE_FORWARD", "STOP", "TURN_LEFT"]}', ["MOVE_FORWARD", "STOP"]),
    ('Thought: the target is behind me.\nActions: TURN_LEFT TURN_LEFT TURN_LEFT STOP',
     ["TURN_LEFT", "TURN_LEFT", "TURN_LEFT", "STOP"]),
    ('{"actions": ["JUMP", "FLY"]}', []),
    ('i am not sure what to do here', []),
]
for text, expected in CASES:
    check(f"parse {text[:32]!r}", parse_actions(text, remaining=500)[0], expected)

check("budget truncation",
      parse_actions('{"actions": ["MOVE_FORWARD","MOVE_FORWARD","MOVE_FORWARD"]}', 2)[0],
      ["MOVE_FORWARD", "MOVE_FORWARD"])

# Guardrail 1: blind-forward cap.
blind = '{"actions": ["MOVE_FORWARD","MOVE_FORWARD","MOVE_FORWARD","MOVE_FORWARD","MOVE_FORWARD","MOVE_FORWARD"]}'
check("forward cap 4 of 6", parse_actions(blind, 500, 8, max_consecutive_forward=4)[0], ["MOVE_FORWARD"] * 4)
check("cap resets on a turn",
      parse_actions('{"actions": ["MOVE_FORWARD","MOVE_FORWARD","TURN_LEFT","MOVE_FORWARD","MOVE_FORWARD"]}',
                    500, 8, max_consecutive_forward=2)[0],
      ["MOVE_FORWARD", "MOVE_FORWARD", "TURN_LEFT", "MOVE_FORWARD", "MOVE_FORWARD"])
check("cap cuts before an unverified STOP",
      parse_actions('{"actions": ["MOVE_FORWARD","MOVE_FORWARD","MOVE_FORWARD","STOP"]}',
                    500, 8, max_consecutive_forward=2)[0], ["MOVE_FORWARD", "MOVE_FORWARD"])


class StubBackend:
    """Stands in for the live endpoint so prompt assembly can be inspected."""

    def __init__(self, replies):
        self.replies = replies if isinstance(replies, list) else [replies]
        self.seen = []
        self.stats = {"calls": 0}

    def infer(self, messages):
        self.seen.append(messages)
        self.stats["calls"] += 1
        return {"text": self.replies[min(len(self.seen) - 1, len(self.replies) - 1)],
                "reasoning_chars": 0, "attempts": 1, "latency_s": 1.0,
                "prompt_tokens": 100, "completion_tokens": 20}


def fake_frame(seed):
    return np.random.default_rng(seed).integers(0, 255, (512, 512, 3), dtype="uint8")


# Guardrail 2: the first STOP is traded for a tilt sweep plus a confirmation turn.
stub = StubBackend(['{"thought": "arrived", "actions": ["STOP"]}'])
policy = VLMPolicy(stub, history_frames=4, max_program=8, verify_stop=True)
policy.reset()
first = policy.decide("go to the door", fake_frame(0), used=10, limit=500)
check("STOP deferred to tilt sweep", first["actions"], ["LOOK_DOWN", "LOOK_UP"])
policy.note(fake_frame(1), first["actions"])
second = policy.decide("go to the door", fake_frame(1), used=12, limit=500)
check("STOP honoured on confirmation", second["actions"], ["STOP"])
check("deferred exactly once per episode", policy.stop_deferrals, 1)
check("confirmation note reached the model", "pitched down" in stub.seen[-1][1]["content"][0]["text"], True)

# Real prompt assembly.
stub2 = StubBackend(['{"thought": "advance", "actions": ["MOVE_FORWARD", "MOVE_FORWARD"]}'])
policy2 = VLMPolicy(stub2, history_frames=4, max_program=8)
policy2.reset()
instruction = "Walk down the hallway and stop by the stairs on your left."
for step in range(6):
    dec = policy2.decide(instruction, fake_frame(step), used=step * 2, limit=500)
    policy2.note(fake_frame(step), dec["actions"])
system, user = stub2.seen[-1]
user_text = user["content"][0]["text"]
check("4 frames attached", sum(1 for b in user["content"] if b["type"] == "image_url"), 4)
check("system template rendered", "{{" not in system["content"] and "{" in system["content"], True)
check("forward cap stated in system prompt", "at most 4 MOVE_FORWARD" in system["content"], True)
check("instruction echoed", instruction in user_text, True)
check("action trace compacted", "FFFF" in user_text, True)
check("budget reported", "of 500 actions used" in user_text, True)

print("\nRESULT:", "PASS" if fails == 0 else f"{fails} FAILURES")
sys.exit(1 if fails else 0)
