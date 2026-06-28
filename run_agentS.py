import argparse
import hashlib
import json
import logging
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/agent-s-replication"))

from modules.manager import Manager
from memory.memory_store import NarrativeMemory, EpisodicMemory
from desktop_env.desktop_env import DesktopEnv
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("AgentS")
client = OpenAI()

SYSTEM = (
    "You control a Linux desktop via pyautogui. "
    "Output EXACTLY ONE line of Python. No imports needed. No comments. No explanations.\n"
    "Valid output examples:\n"
    "pyautogui.click(x=856, y=44)\n"
    "pyautogui.hotkey('ctrl', 's')\n"
    "pyautogui.typewrite('hello world', interval=0.05)\n"
    "pyautogui.press('enter')\n"
    "pyautogui.scroll(0, -3)\n"
    "DONE\n"
    "FAIL"
)


def get_action(task, subtask_name, subtask_desc, a11y, history):
    recent = "\n".join(history[-5:]) or "none"
    user_msg = (
        "Task: " + task + "\n"
        "Subtask: " + subtask_name + " - " + subtask_desc + "\n"
        "Recent actions:\n" + recent + "\n\n"
        "Accessibility tree:\n" + a11y[:4000]
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=80,
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_msg}
        ]
    )
    return resp.choices[0].message.content.strip()


def tree_hash(t):
    return hashlib.md5(t.encode()).hexdigest()


def run(args):
    with open(args.test_all_meta_path) as f:
        meta = json.load(f)

    Mn = NarrativeMemory("narrative_memory.pkl")
    Me = EpisodicMemory("episodic_memory.pkl")
    manager = Manager(Mn)

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    env = DesktopEnv(
        provider_name=args.provider_name,
        headless=args.headless,
        action_space="pyautogui",
        require_a11y_tree=True,
    )

    scores = []

    for domain, ids in meta.items():
        for eid in ids:
            cfg = "evaluation_examples/examples/" + domain + "/" + eid + ".json"
            if not os.path.exists(cfg):
                logger.warning("Config not found: " + cfg)
                continue

            with open(cfg) as f:
                example = json.load(f)

            instruction = example.get("instruction", "")
            logger.info("\n[" + domain + "/" + eid + "] " + instruction[:80])

            score = 0.0
            try:
                obs = env.reset(task_config=example)
                total = 0

                try:
                    _, subtasks = manager.run(instruction, obs.get("accessibility_tree", ""))
                    logger.info("Planned " + str(len(subtasks)) + " subtasks")
                except Exception as e:
                    logger.error("Manager error: " + str(e))
                    subtasks = [{"name": "complete_task", "description": instruction}]

                for subtask in subtasks:
                    if total >= args.max_steps:
                        break

                    sname = subtask.get("name", "subtask")
                    sdesc = subtask.get("description", instruction)
                    history = []
                    last = ""
                    repeats = 0
                    hashes = []

                    for _ in range(args.max_steps - total):
                        a11y = obs.get("accessibility_tree", "")

                        hashes.append(tree_hash(a11y))
                        if len(hashes) >= 3 and len(set(hashes[-3:])) == 1:
                            logger.info("Stagnation: tree unchanged 3 steps")
                            break

                        action = get_action(instruction, sname, sdesc, a11y, history)
                        logger.info("  Step " + str(total + 1) + ": " + action[:80])

                        if action in ("DONE", "FAIL"):
                            break

                        if action == last:
                            repeats += 1
                            if repeats >= 2:
                                logger.info("Stagnation: same action repeated")
                                break
                        else:
                            repeats = 0
                        last = action

                        cmd = action if action.startswith("import") else "import pyautogui; " + action

                        try:
                            obs, reward, done, info = env.step(cmd)
                            history.append(action)
                            total += 1
                            if done:
                                break
                        except Exception as e:
                            logger.error("Step error: " + str(e))
                            break

                score = float(env.evaluate())

            except Exception as e:
                logger.error("Task error: " + str(e))
                traceback.print_exc()

            scores.append(score)
            logger.info("Score: " + str(score))
            print("Logged result: " + domain + "/" + eid + " -> score: " + str(score), flush=True)

            with open(result_dir / (domain + "_" + eid + ".json"), "w") as f:
                json.dump({"id": eid, "domain": domain, "score": score}, f)

    avg = sum(scores) / len(scores) if scores else 0.0
    print("Average score: " + str(round(avg, 4)), flush=True)

    with open(result_dir / "summary.json", "w") as f:
        json.dump({"average_score": avg, "n_tasks": len(scores)}, f)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--provider_name", default="docker")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--test_all_meta_path", default="evaluation_examples/test_small.json")
    p.add_argument("--max_steps", type=int, default=15)
    p.add_argument("--result_dir", default="./results_agentS3")
    run(p.parse_args())
