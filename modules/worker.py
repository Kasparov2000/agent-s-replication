"""
Worker Module for Agent S.
Responsibilities:
  - Retrieve episodic memory Me for current subtask.
  - Run step loop (max 15 steps): observe → reflect → generate action → execute via ACI.
  - Trajectory Reflector with stagnation detector (Novel Improvement B).
  - Grounding Verifier integration (Novel Improvement A).
  - Signal DONE, FAIL, or PARTIAL_FAIL.
"""

import os
import hashlib
from dataclasses import dataclass, field
from openai import OpenAI
from memory.memory_store import EpisodicMemory
from aci.interface import ACIExecutor, UIElement
from modules.grounding_verifier import verify_grounding

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
MODEL = "gpt-4o"
MAX_STEPS = 15
STAGNATION_REPEAT_LIMIT = 2   # same action N times → PARTIAL_FAIL
STAGNATION_TREE_LIMIT = 3     # tree unchanged N steps → PARTIAL_FAIL


@dataclass
class StepRecord:
    step: int
    action: str
    obs_summary: str
    reflection: str
    status: str   # "ok" | "done" | "fail" | "partial_fail"


def _gpt(system: str, user: str, max_tokens: int = 1024) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    )
    return resp.choices[0].message.content.strip()


def _tree_hash(obs_tree: str) -> str:
    return hashlib.md5(obs_tree.encode()).hexdigest()


class TrajectoryReflector:
    """
    Provides adaptive guidance to the Action Generator.
    Novel Improvement B: stagnation detector escalates to PARTIAL_FAIL.
    """

    def __init__(self):
        self.action_history: list[str] = []
        self.tree_hashes: list[str] = []

    def update(self, action: str, tree_hash: str):
        self.action_history.append(action)
        self.tree_hashes.append(tree_hash)

    def check_stagnation(self) -> tuple[bool, str]:
        """Returns (is_stagnating, reason)."""
        # Check repeated identical actions
        if len(self.action_history) >= STAGNATION_REPEAT_LIMIT:
            last_n = self.action_history[-STAGNATION_REPEAT_LIMIT:]
            if len(set(last_n)) == 1:
                return True, f"Same action repeated {STAGNATION_REPEAT_LIMIT} times: {last_n[0]}"

        # Check frozen UI state
        if len(self.tree_hashes) >= STAGNATION_TREE_LIMIT:
            last_n = self.tree_hashes[-STAGNATION_TREE_LIMIT:]
            if len(set(last_n)) == 1:
                return True, f"Accessibility tree unchanged for {STAGNATION_TREE_LIMIT} consecutive steps."

        return False, ""

    def advise(self, subtask: str, trajectory_summary: str) -> str:
        """Ask GPT-4o for reflective guidance given current trajectory."""
        system = (
            "You are a trajectory reflector for a GUI automation agent. "
            "Review the action history and suggest an alternative strategy if the agent appears stuck. "
            "Be concise and specific. If progress looks good, say 'Continue as planned.'"
        )
        user = f"Subtask: {subtask}\n\nAction history:\n{trajectory_summary}"
        return _gpt(system, user, max_tokens=200)


class Worker:
    def __init__(
        self,
        task: str,
        subtask: dict,
        episodic_memory: EpisodicMemory,
        aci_executor: ACIExecutor,
        screenshot_fn,    # callable() -> path to current screenshot
        tree_fn,          # callable() -> (elements list, linearized tree str)
    ):
        self.task = task
        self.subtask = subtask
        self.Me = episodic_memory
        self.aci = aci_executor
        self.get_screenshot = screenshot_fn
        self.get_tree = tree_fn
        self.reflector = TrajectoryReflector()
        self.step_log: list[StepRecord] = []

    def _retrieve_episodic(self, app_state: dict) -> str:
        entries = self.Me.retrieve(
            task=self.task,
            subtask=self.subtask["name"],
            context=self.subtask.get("context", ""),
            app_state=app_state,
            k=3,
        )
        if not entries:
            return "No relevant past experience found."
        return "\n\n".join([f"Example {i+1}:\n{e.summary}" for i, e in enumerate(entries)])

    def _generate_action(
        self,
        obs_tree: str,
        episodic_context: str,
        reflector_advice: str,
        grounding_feedback: str = "",
    ) -> dict:
        """
        Chain-of-thought four-part response:
          1. Previous action status
          2. Observation analysis
          3. Semantic next action description
          4. Grounded next action (agent.primitive(...))
        """
        system = (
            "You are a GUI automation agent executing a subtask. "
            "Respond with exactly four labeled sections:\n"
            "1. PREVIOUS_ACTION_STATUS: Was the last action successful? What changed?\n"
            "2. OBSERVATION_ANALYSIS: What is the current GUI state relevant to the subtask?\n"
            "3. SEMANTIC_NEXT_ACTION: Describe the next logical action in plain English.\n"
            "4. GROUNDED_NEXT_ACTION: One action call, e.g. agent.click(41, 1, 'left') or agent.done()\n\n"
            "Available primitives:\n"
            "  agent.click(eid, num_clicks, button, hold_keys=[])\n"
            "  agent.type(text, eid=None)\n"
            "  agent.scroll(eid, direction)\n"
            "  agent.hotkey([keys])\n"
            "  agent.hold_and_press(hold, [press_keys])\n"
            "  agent.drag_and_drop(src_eid, dst_eid)\n"
            "  agent.save_to_buffer(eid)\n"
            "  agent.switch_applications(app_name)\n"
            "  agent.wait(seconds)\n"
            "  agent.done()\n"
            "  agent.fail()\n"
        )

        prev_actions = "\n".join(
            [f"Step {r.step}: {r.action} → {r.status}" for r in self.step_log[-5:]]
        ) or "None yet."

        grounding_note = f"\n[GROUNDING FEEDBACK]: {grounding_feedback}" if grounding_feedback else ""

        user = (
            f"Task: {self.task}\n"
            f"Subtask: {self.subtask['name']} — {self.subtask['description']}\n\n"
            f"Past Experience:\n{episodic_context}\n\n"
            f"Reflector Advice: {reflector_advice}\n"
            f"{grounding_note}\n"
            f"Recent Actions:\n{prev_actions}\n\n"
            f"Current Accessibility Tree:\n{obs_tree[:3000]}"
        )

        raw = _gpt(system, user, max_tokens=600)

        # Parse the four parts
        parts = {}
        for label in ["PREVIOUS_ACTION_STATUS", "OBSERVATION_ANALYSIS", "SEMANTIC_NEXT_ACTION", "GROUNDED_NEXT_ACTION"]:
            import re
            m = re.search(rf"{label}:(.*?)(?=(?:PREVIOUS_ACTION_STATUS|OBSERVATION_ANALYSIS|SEMANTIC_NEXT_ACTION|GROUNDED_NEXT_ACTION):|$)", raw, re.DOTALL)
            parts[label] = m.group(1).strip() if m else ""

        return parts

    def run(self) -> tuple[str, list[StepRecord]]:
        """
        Execute the subtask. Returns ("DONE" | "FAIL" | "PARTIAL_FAIL", step_log).
        """
        grounding_feedback = ""

        for step in range(1, MAX_STEPS + 1):
            # Observe
            screenshot_path = self.get_screenshot()
            elements, obs_tree = self.get_tree()
            self.aci.update_elements(elements)

            from aci.interface import extract_app_state
            app_state = extract_app_state(elements)
            tree_h = _tree_hash(obs_tree)

            # Retrieve episodic memory on first step only (cost saving)
            if step == 1:
                episodic_context = self._retrieve_episodic(app_state)
            
            # Reflector advice
            traj_summary = "\n".join([f"Step {r.step}: {r.action}" for r in self.step_log])
            reflector_advice = self.reflector.advise(self.subtask["name"], traj_summary) if self.step_log else "Starting subtask."

            # Generate action
            parts = self._generate_action(obs_tree, episodic_context, reflector_advice, grounding_feedback)
            action_str = parts.get("GROUNDED_NEXT_ACTION", "agent.fail()")
            semantic_intent = parts.get("SEMANTIC_NEXT_ACTION", "")

            # Grounding Verifier (Novel Improvement A)
            gv_ok, grounding_feedback = verify_grounding(
                action_str, semantic_intent, screenshot_path, self.aci._element_map
            )
            if not gv_ok:
                print(f"[Worker] Step {step}: Grounding rejected — {grounding_feedback}")
                self.step_log.append(StepRecord(step, action_str, obs_tree[:200], grounding_feedback, "grounding_fail"))
                continue  # retry with feedback injected next iteration

            # Execute
            result = self.aci.dispatch(action_str)
            grounding_feedback = ""  # reset on successful dispatch

            # Update reflector state
            self.reflector.update(action_str, tree_h)

            status = "ok"
            if result == "DONE":
                status = "done"
                self.step_log.append(StepRecord(step, action_str, obs_tree[:200], reflector_advice, status))
                return "DONE", self.step_log
            elif result == "FAIL":
                status = "fail"
                self.step_log.append(StepRecord(step, action_str, obs_tree[:200], reflector_advice, status))
                return "FAIL", self.step_log

            self.step_log.append(StepRecord(step, action_str, obs_tree[:200], reflector_advice, status))

            # Stagnation check (Novel Improvement B)
            is_stagnating, stagnation_reason = self.reflector.check_stagnation()
            if is_stagnating:
                print(f"[Worker] Stagnation detected: {stagnation_reason}")
                return "PARTIAL_FAIL", self.step_log

        # Exceeded max steps
        return "FAIL", self.step_log
