"""
OSWorld Evaluation Runner for Agent S.
Runs the full 369-task evaluation and per-category breakdown.
Expects OSWorld environment with functional evaluation scripts.

Usage:
    python evaluation/osworld_eval.py \
        --tasks_dir /path/to/osworld/test \
        --results_dir ./results \
        --max_steps 50
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from agent import AgentS

CATEGORIES = ["os", "office", "daily", "professional", "workflow"]


def load_tasks(tasks_dir: str) -> list[dict]:
    """Load all OSWorld test tasks from JSON files."""
    tasks = []
    for path in Path(tasks_dir).rglob("*.json"):
        with open(path) as f:
            task = json.load(f)
        task["_path"] = str(path)
        tasks.append(task)
    return tasks


def get_category(task: dict) -> str:
    """Infer category from task metadata or path."""
    cat = task.get("category", "").lower()
    if cat in CATEGORIES:
        return cat
    # Infer from path
    path = task.get("_path", "")
    for c in CATEGORIES:
        if c in path.lower():
            return c
    return "unknown"


def restore_snapshot(snapshot_id: str, osworld_script: str = "reset_env.sh"):
    """Restore the OSWorld VM snapshot before each task."""
    subprocess.run([osworld_script, snapshot_id], check=False, timeout=60)
    time.sleep(3)  # wait for VM to settle


def run_functional_eval(task: dict, eval_script: str = "evaluate.py") -> bool:
    """Run OSWorld's functional evaluation script for a task. Returns True if passed."""
    task_id = task.get("id", "unknown")
    result = subprocess.run(
        ["python", eval_script, "--task_id", task_id],
        capture_output=True, text=True, timeout=120
    )
    return "SUCCESS" in result.stdout or result.returncode == 0


def evaluate(
    tasks_dir: str,
    results_dir: str,
    max_steps: int = 50,
    osworld_eval_script: str = "evaluate.py",
    reset_script: str = "reset_env.sh",
    task_subset: list[str] = None,  # list of task IDs; None = all
):
    os.makedirs(results_dir, exist_ok=True)

    agent = AgentS(
        narrative_memory_path=os.path.join(results_dir, "narrative_memory.pkl"),
        episodic_memory_path=os.path.join(results_dir, "episodic_memory.pkl"),
        max_total_steps=max_steps,
    )

    tasks = load_tasks(tasks_dir)
    if task_subset:
        tasks = [t for t in tasks if t.get("id") in task_subset]

    print(f"[Eval] Running {len(tasks)} tasks...")

    results = []
    category_counts = {c: {"total": 0, "success": 0} for c in CATEGORIES + ["unknown"]}

    for i, task in enumerate(tasks):
        task_id = task.get("id", f"task_{i}")
        task_desc = task.get("instruction", task.get("task", ""))
        category = get_category(task)
        snapshot_id = task.get("snapshot_id", task_id)

        print(f"\n[Eval] Task {i+1}/{len(tasks)} [{category}]: {task_desc[:80]}")

        # Restore VM snapshot
        restore_snapshot(snapshot_id, reset_script)

        # Run agent
        try:
            agent_success = agent.run(task_desc)
        except Exception as ex:
            print(f"[Eval] Agent error: {ex}")
            agent_success = False

        # Functional evaluation
        func_success = run_functional_eval(task, osworld_eval_script) if agent_success else False

        record = {
            "task_id": task_id,
            "category": category,
            "task": task_desc,
            "agent_success": agent_success,
            "func_success": func_success,
        }
        results.append(record)

        cat = category if category in category_counts else "unknown"
        category_counts[cat]["total"] += 1
        if func_success:
            category_counts[cat]["success"] += 1

        # Save incremental results
        with open(os.path.join(results_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    total_tasks = len(results)
    total_success = sum(1 for r in results if r["func_success"])
    print(f"Overall: {total_success}/{total_tasks} = {100*total_success/max(total_tasks,1):.2f}%\n")

    for cat, counts in category_counts.items():
        if counts["total"] > 0:
            pct = 100 * counts["success"] / counts["total"]
            print(f"  {cat:15s}: {counts['success']}/{counts['total']} = {pct:.2f}%")

    # Save summary
    summary = {
        "overall": {"success": total_success, "total": total_tasks},
        "by_category": category_counts,
    }
    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {results_dir}/")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks_dir", required=True)
    parser.add_argument("--results_dir", default="./results")
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--eval_script", default="evaluate.py")
    parser.add_argument("--reset_script", default="reset_env.sh")
    parser.add_argument("--subset", nargs="*", help="Task IDs to evaluate (optional)")
    args = parser.parse_args()

    evaluate(
        tasks_dir=args.tasks_dir,
        results_dir=args.results_dir,
        max_steps=args.max_steps,
        osworld_eval_script=args.eval_script,
        reset_script=args.reset_script,
        task_subset=args.subset,
    )
