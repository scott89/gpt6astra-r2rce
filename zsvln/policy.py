"""Prompting, action parsing, and the two decision policies."""
from __future__ import annotations

import json
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from zsvln.backend import text_block, image_block

VOCAB = ("STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "LOOK_UP", "LOOK_DOWN")
SHORT = {
    "STOP": "S",
    "MOVE_FORWARD": "F",
    "TURN_LEFT": "L",
    "TURN_RIGHT": "R",
    "LOOK_UP": "U",
    "LOOK_DOWN": "D",
}

SYSTEM_TMPL = """You are an embodied navigation agent inside a photorealistic house, running the Room-to-Room (R2R) Vision-and-Language Navigation in Continuous Environments benchmark.

You control a robot of 0.88 m eye height equipped with ONE monocular camera. You have NO map, NO depth sensor, NO GPS/localization, and NO sensor of your distance to the target. The images below are the only observations you ever get.

Following the instruction, reach the location it describes and then STOP.

Action vocabulary -- each name is exactly one primitive action:
  MOVE_FORWARD : advance 0.25 m in the direction you face
  TURN_LEFT    : rotate the whole robot 15 degrees left
  TURN_RIGHT   : rotate the whole robot 15 degrees right
  LOOK_UP      : pitch the camera up 30 degrees (robot does not move)
  LOOK_DOWN    : pitch the camera down 30 degrees (robot does not move)
  STOP         : end the episode here. The episode counts as SUCCESS only if you explicitly issue STOP while within 3 m of the goal; running out of the action budget without STOPPING is always a failure.

Respond with a single JSON object and nothing else:
  {{"thought": "<why, 1-2 sentences>", "actions": ["<ACTION>", ...]}}
`actions` is the program you want executed before your next camera frame. Never emit an action name outside the vocabulary.
Rules that keep you alive:
- MOVE_FORWARD is blind. Chain at most {max_fwd} MOVE_FORWARD in a single program, then re-observe before committing to more travel.
- Re-aim with TURN_LEFT/TURN_RIGHT and check alignment before moving into a doorway; a forward step through a badly-aimed doorway means a collision and a lost sense of direction.
- If you propose STOP, the camera will first pitch down and back up so you get one fresh look at the spot, and you will be asked to confirm. Only STOP at the place the instruction actually ends at -- stopping 3-5 m short or 3-5 m past it both count as failure."""

USER_TMPL = """Instruction: {instruction}

Action budget: {used} of {limit} actions used, {remaining} remaining.
Actions executed so far ({trace_len} primitives):
{trace}

{frames_note}"""


def _json_object(text: str) -> Optional[Dict[str, Any]]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*", "", cleaned).strip().strip("`").strip()
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except Exception:  # noqa: BLE001
        pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        for candidate in (match.group(),):
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    return obj
            except Exception:  # noqa: BLE001
                continue
    return None


def _vocab_tokens(text: str) -> List[str]:
    found: List[Tuple[int, str]] = []
    for name in VOCAB:
        for match in re.finditer(name, text.upper()):
            found.append((match.start(), name))
    return [name for _, name in sorted(found)]


def parse_actions(text: str, remaining: int, max_program: int = 8,
                  max_consecutive_forward: int = 0) -> Tuple[List[str], str, bool]:
    """Return (actions, thought, parsed_ok). Actions are budget-truncated and STOP-terminated."""
    obj = _json_object(text or "")
    thought = ""
    raw: Any = None
    parsed = False
    if obj is not None:
        parsed = True
        thought = str(obj.get("thought", obj.get("reasoning", "")))[:1200]
        raw = obj.get("actions", obj.get("action"))

    actions: List[str] = []
    if isinstance(raw, str):
        raw = [raw]
    if isinstance(raw, list):
        for item in raw:
            name = str(item).strip().upper().replace(" ", "_").replace("-", "_")
            name = {
                "F": "MOVE_FORWARD", "FORWARD": "MOVE_FORWARD", "MOVE": "MOVE_FORWARD",
                "L": "TURN_LEFT", "LEFT": "TURN_LEFT",
                "R": "TURN_RIGHT", "RIGHT": "TURN_RIGHT",
                "U": "LOOK_UP", "D": "LOOK_DOWN",
            }.get(name, name)
            if name in VOCAB:
                actions.append(name)
    if not actions and not parsed:
        actions = _vocab_tokens(text or "")[:max_program]
        parsed = bool(actions)

    actions = actions[:max_program]
    if "STOP" in actions:
        actions = actions[: actions.index("STOP") + 1]
    if max_consecutive_forward > 0:
        # Guardrail against blind dead-reckoning: cut the program short so the
        # agent must re-observe rather than chain forwards it cannot verify.
        capped: List[str] = []
        run = 0
        for name in actions:
            if name == "MOVE_FORWARD":
                run += 1
                if run > max_consecutive_forward:
                    break
            else:
                run = 0
            capped.append(name)
        actions = capped
    actions = actions[: max(0, remaining)]
    return actions, thought, parsed and bool(actions)


class VLMPolicy:
    """Reproduces the paper's agent: the VLM emits a primitive program, then re-observes."""

    name = "vlm"

    def __init__(self, backend, history_frames: int = 4, max_program: int = 8,
                 max_consecutive_forward: int = 4, verify_stop: bool = True) -> None:
        self.backend = backend
        self.history_frames = max(1, min(history_frames, 8))
        self.max_program = max_program
        self.max_consecutive_forward = max_consecutive_forward
        self.verify_stop = verify_stop
        self.reset()

    def reset(self) -> None:
        self.trace: List[str] = []
        self.frames: List[Any] = []
        self.calls: int = 0
        self.parse_failures: int = 0
        self.stop_pending = False
        self.stop_verified_once = False
        self.stop_deferrals = 0

    def note(self, rgb: Optional[Any], executed: List[str]) -> None:
        self.trace.extend(executed)
        if rgb is not None:
            self.frames.append(rgb.copy())
            self.frames = self.frames[-(self.history_frames + 1):]

    def _trace_text(self) -> str:
        if not self.trace:
            return "(none yet -- you are at the start position)"
        return "".join(SHORT[a] for a in self.trace)

    def decide(self, instruction: str, rgb: Any, used: int, limit: int) -> Dict[str, Any]:
        remaining = max(0, limit - used)
        frames = (self.frames + [rgb])[-self.history_frames:] if self.frames else [rgb]
        frames_note = (
            f"The last {len(frames)} camera frames, oldest first; the final image is what you see right now."
        )
        if self.stop_pending:
            frames_note = (
                "You proposed STOP at the previous decision. Since then the camera pitched down and "
                "back to level at that same spot, so the final image is a fresh look at where you are. "
                "Confirm STOP only if this really is the destination the instruction ends at; "
                "otherwise keep navigating.\n\n" + frames_note
            )
            self.stop_pending = False
        content: List[Dict[str, Any]] = [
            {
                "type": "text",
                "text": USER_TMPL.format(
                    instruction=instruction.strip(),
                    used=used,
                    limit=limit,
                    remaining=remaining,
                    trace_len=len(self.trace),
                    trace=self._trace_text(),
                    frames_note=frames_note,
                ),
            }
        ]
        for frame in frames:
            content.append(image_block(frame))

        messages = [
            {"role": "system", "content": SYSTEM_TMPL.format(max_fwd=self.max_consecutive_forward)},
            {"role": "user", "content": content},
        ]

        last_text, actions, thought, ok = "", [], "", False
        latency, prompt_tokens, completion_tokens = 0.0, 0, 0
        for attempt in range(2):
            resp = self.backend.infer(messages)
            self.calls += 1
            last_text = resp["text"]
            latency += float(resp.get("latency_s", 0.0))
            prompt_tokens += int(resp.get("prompt_tokens", 0))
            completion_tokens += int(resp.get("completion_tokens", 0))
            actions, thought, ok = parse_actions(
                last_text, remaining, self.max_program, self.max_consecutive_forward
            )
            if ok:
                break
            self.parse_failures += 1
            messages.append({"role": "assistant", "content": last_text[:2000]})
            messages.append(
                {
                    "role": "user",
                    "content": [
                        text_block(
                            "That reply contained no usable action. Reply with only the JSON object "
                            '{"thought": "...", "actions": ["MOVE_FORWARD"]} using names from the vocabulary.'
                        )
                    ],
                }
            )

        # Guardrail: look before you leap. Trade the first STOP for a tilt sweep so
        # the agent gets one more observation before committing to the goal test.
        if (self.verify_stop and actions and actions[-1] == "STOP"
                and not self.stop_verified_once and remaining > 3):
            self.stop_verified_once = True
            self.stop_pending = True
            self.stop_deferrals += 1
            actions = ["LOOK_DOWN", "LOOK_UP"]

        if not actions:
            # Never stall an episode on a malformed reply: spend the budget moving, then stop.
            actions = ["STOP"] if remaining <= 1 else ["MOVE_FORWARD"]
        return {
            "actions": actions,
            "thought": thought,
            "raw": last_text,
            "parsed_ok": ok,
            "latency_s": round(latency, 2),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }


class ScriptedPolicy:
    """No-network stand-in used to validate the harness, budgets and metrics."""

    name = "scripted"

    def __init__(self, seed: int = 0, stop_after: int = 60, max_program: int = 4) -> None:
        self.rng = random.Random(seed)
        self.stop_after = stop_after
        self.max_program = max_program
        self.reset()

    def reset(self) -> None:
        self.trace: List[str] = []
        self.calls = 0
        self.parse_failures = 0

    def note(self, rgb: Optional[Any], executed: List[str]) -> None:
        self.trace.extend(executed)

    def decide(self, instruction: str, rgb: Any, used: int, limit: int) -> Dict[str, Any]:
        self.calls += 1
        remaining = max(0, limit - used)
        if used >= self.stop_after or remaining <= 1:
            actions = ["STOP"]
        else:
            actions = []
            for _ in range(min(self.max_program, remaining - 1)):
                roll = self.rng.random()
                actions.append("MOVE_FORWARD" if roll < 0.7 else ("TURN_LEFT" if roll < 0.85 else "TURN_RIGHT"))
        return {"actions": actions, "thought": "scripted", "raw": json.dumps(actions), "parsed_ok": True,
                "latency_s": 0.0, "prompt_tokens": 0, "completion_tokens": 0}
