"""
Agent S — Main Orchestrator
Ties together Manager, Worker, Self-Evaluator, ACI, and Memory.

Usage:
    python agent.py --task "Open LibreOffice Calc and sum column A" --max_steps 50
"""

import argparse
import os
import time
import subprocess
import tempfile
from pathlib import Path

from memory.memory_store import NarrativeMemory, EpisodicMemory
from aci.interface import ACIExecutor, get_accessibility_tree, ocr_augment_tree, linearize_tree, extract_app_state
from modules.manager import Manager
from modules.worker import Worker
from modules.self_evaluator import SelfEvaluator

# ─────────────────────────────────────────────
# Screenshot helper
# ─────────────────────────────────────────────

def take_screenshot() -> str:
    """Capture the current desktop to a temp PNG file. Returns path."""
    path = os.path.join(tempfile.gettempdir(), f"agentS_screen_{int(time.time()*1000)}.png")
    try:
        import pyautogui
        pyautogui.screenshot(path)
    except Exception:
        # Fallback: scrot (Linux)
        subprocess.run(["scrot", path], check=False)
    return path


def get_obs() -> tuple[list, str]:
    """Return (elements, linearized_tree_str) with OCR augmentation."""
    screenshot = take_screenshot()
    elements = get_accessibility_tree()
    elements = ocr_augment_tree(elements, screenshot)
    tree_str = linearize_tree(elements)
    return elements, tree_str


# ─────────────────────────────────────────────
# AgentS
# ─────────────────────────────────────────────

class AgentS:
    def __init__(
        self,
        narrative_memory_path: str = "narrative_memory.pkl",
        episodic_memory_path: str = "episodic_memory.pkl",
        max_total_steps: int = 50,
    ):
        self.Mn = NarrativeMemory(narrative_memory_path)
        self.Me = EpisodicMemory(episodic_memory_path)
        self.aci = ACIExecutor()
        self.manager = Manager(self.Mn)
        self.evaluator = SelfEvaluator(self.Mn, self.Me)
        self.max_total_steps = max_total_steps

    def run(self, task: str) -> bool:
        """
        Execute a task end-to-end.
        Returns True if the task completed successfully.
        """
        print(f"\n{'='*60}")
        print(f"[AgentS] Task: {task}")
        print(f"{'='*60}")

        # Initial observation
        elements, obs_tree = get_obs()
        app_state = extract_app_state(elements)

        # Manager: plan subtasks
        Q, subtask_queue = self.manager.run(task, obs_tree)
        print(f"[Manager] Query: {Q}")
        print(f"[Manager] Planned {len(subtask_queue)} subtasks:")
        for i, s in enumerate(subtask_queue):
            print(f"  {i+1}. {s['name']}: {s['description']}")

        completed_subtasks = []
        subtask_summaries = []
        total_steps = 0
        final_obs_tree = obs_tree

        subtask_idx = 0
        while subtask_idx < len(subtask_queue) and total_steps < self.max_total_steps:
            subtask = subtask_queue[subtask_idx]
            print(f"\n[Worker] Executing subtask {subtask_idx+1}/{len(subtask_queue)}: {subtask['name']}")

            worker = Worker(
                task=task,
                subtask=subtask,
                episodic_memory=self.Me,
                aci_executor=self.aci,
                screenshot_fn=take_screenshot,
                tree_fn=get_obs,
            )

            result, step_log = worker.run()
            total_steps += len(step_log)

            # Get current obs for app state
            elements, final_obs_tree = get_obs()
            current_app_state = extract_app_state(elements)

            print(f"[Worker] Subtask result: {result} ({len(step_log)} steps)")

            if result == "DONE":
                # Self-evaluator: summarize subtask and store in Me
                summary = self.evaluator.summarize_subtask(task, subtask, step_log, current_app_state)
                subtask_summaries.append(summary)
                completed_subtasks.append(subtask)
                subtask_idx += 1

            elif result == "PARTIAL_FAIL":
                # Novel Improvement B: replan from current state
                print(f"[Manager] Stagnation — replanning from current state...")
                stagnation_reason = "Stagnation detected (repeated actions or frozen UI)."
                new_queue = self.manager.replan(
                    task=task,
                    current_obs_tree=final_obs_tree,
                    completed=completed_subtasks,
                    failed_subtask=subtask,
                    reason=stagnation_reason,
                )
                # Replace remaining subtasks with replanned ones
                subtask_queue = completed_subtasks + new_queue
                subtask_idx = len(completed_subtasks)
                print(f"[Manager] Replanned {len(new_queue)} remaining subtasks.")

            else:  # FAIL
                print(f"[Manager] Subtask FAIL — replanning...")
                new_queue = self.manager.replan(
                    task=task,
                    current_obs_tree=final_obs_tree,
                    completed=completed_subtasks,
                    failed_subtask=subtask,
                    reason="Worker signaled FAIL after exhausting steps.",
                )
                if not new_queue:
                    print("[AgentS] No replan possible. Task FAILED.")
                    break
                subtask_queue = completed_subtasks + new_queue
                subtask_idx = len(completed_subtasks)

        # Determine success: all subtasks completed
        success = subtask_idx >= len(subtask_queue) and total_steps <= self.max_total_steps
        print(f"\n[AgentS] Task {'SUCCEEDED' if success else 'FAILED'} "
              f"({subtask_idx}/{len(subtask_queue)} subtasks, {total_steps} total steps)")

        # Self-evaluator: store narrative summary
        self.evaluator.summarize_task(
            task=task,
            query=Q,
            subtask_summaries=subtask_summaries,
            success=success,
            final_obs_tree=final_obs_tree,
        )

        return success


# ─────────────────────────────────────────────
# Memory Bootstrapping (exploration phase)
# ─────────────────────────────────────────────

def bootstrap_memory(task_list: list[str], agent: AgentS):
    """
    Run Agent S on exploration tasks to populate memory before evaluation.
    Web knowledge only — no memory retrieval during bootstrap.
    """
    print(f"[Bootstrap] Starting exploration on {len(task_list)} tasks...")
    for i, task in enumerate(task_list):
        print(f"\n[Bootstrap] Task {i+1}/{len(task_list)}: {task}")
        try:
            agent.run(task)
        except Exception as ex:
            print(f"[Bootstrap] Error on task '{task}': {ex}")
    print("[Bootstrap] Done.")


# ─────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agent S — GUI Automation Agent")
    parser.add_argument("--task", type=str, required=True, help="Task description")
    parser.add_argument("--max_steps", type=int, default=50, help="Max total steps")
    parser.add_argument(
        "--narrative_memory", type=str, default="narrative_memory.pkl"
    )
    parser.add_argument(
        "--episodic_memory", type=str, default="episodic_memory.pkl"
    )
    args = parser.parse_args()

    agent = AgentS(
        narrative_memory_path=args.narrative_memory,
        episodic_memory_path=args.episodic_memory,
        max_total_steps=args.max_steps,
    )
    success = agent.run(args.task)
    exit(0 if success else 1)
