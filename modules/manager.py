"""
Manager Module for Agent S.
Responsibilities:
  1. Formulate observation-aware query Q from task + initial GUI state.
  2. Retrieve web knowledge via Perplexica.
  3. Retrieve narrative memory Mn.
  4. Fuse knowledge into K_fused.
  5. Generate topologically sorted subtask queue.
"""

import json
import os
import re
import requests
from openai import OpenAI
from memory.memory_store import NarrativeMemory

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
MODEL = "gpt-4o"
PERPLEXICA_URL = os.environ.get("PERPLEXICA_URL", "http://localhost:3000/api/search")


# ─────────────────────────────────────────────
# Web Search via Perplexica
# ─────────────────────────────────────────────

def web_search(query: str, top_k: int = 3) -> str:
    """Query self-hosted Perplexica and return top results as text."""
    try:
        resp = requests.post(
            PERPLEXICA_URL,
            json={"query": query, "focusMode": "webSearch"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        sources = data.get("sources", [])[:top_k]
        snippets = [f"[{i+1}] {s.get('title','')}: {s.get('snippet','')}" for i, s in enumerate(sources)]
        return "\n".join(snippets) if snippets else "No results found."
    except Exception as ex:
        return f"[WebSearch unavailable: {ex}]"


# ─────────────────────────────────────────────
# GPT-4o Calls
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# Manager
# ─────────────────────────────────────────────

class Manager:
    def __init__(self, narrative_memory: NarrativeMemory):
        self.Mn = narrative_memory

    def formulate_query(self, task: str, obs_tree: str) -> str:
        """Produce an observation-aware query Q in 'How to do X' format."""
        system = (
            "You are the planning component of a GUI automation agent. "
            "Given a user task and the current GUI accessibility tree, "
            "formulate a concise search query in the format 'How to <action> in <application>'."
        )
        user = f"Task: {task}\n\nCurrent GUI state:\n{obs_tree[:2000]}"
        return _gpt(system, user, max_tokens=80)

    def fuse_knowledge(self, query: str, web_knowledge: str, narrative_summaries: list[str]) -> str:
        """Fuse web results and narrative memory into K_fused."""
        system = (
            "You are a knowledge fusion component. Combine web search results and past experience summaries "
            "into a concise knowledge block to guide task planning. "
            "Prefer experience for application-specific behavior; prefer web knowledge for version-specific procedures. "
            "Resolve contradictions explicitly. Be concise."
        )
        narratives_text = "\n\n".join(narrative_summaries) if narrative_summaries else "No past experience available."
        user = (
            f"Query: {query}\n\n"
            f"Web Knowledge:\n{web_knowledge}\n\n"
            f"Past Experience:\n{narratives_text}"
        )
        return _gpt(system, user, max_tokens=512)

    def plan_subtasks(self, task: str, fused_knowledge: str) -> list[dict]:
        """
        Generate a JSON-structured subtask queue.
        Each subtask: {name, description, context}
        """
        system = (
            "You are a task planner for a GUI automation agent. "
            "Given a task and background knowledge, output a JSON array of subtasks in execution order. "
            "Each element must have keys: 'name' (short label), 'description' (what to do), 'context' (relevant UI hints). "
            "Output ONLY the JSON array, no other text."
        )
        user = f"Task: {task}\n\nBackground Knowledge:\n{fused_knowledge}"
        raw = _gpt(system, user, max_tokens=800)

        # Parse JSON robustly
        raw = re.sub(r"```json|```", "", raw).strip()
        try:
            subtasks = json.loads(raw)
            assert isinstance(subtasks, list)
            return subtasks
        except Exception:
            # Fallback: single subtask wrapping the whole task
            return [{"name": "Execute Task", "description": task, "context": ""}]

    def run(self, task: str, obs_tree: str) -> tuple[str, list[dict]]:
        """
        Full Manager pipeline.
        Returns (query Q, subtask_queue).
        """
        # Step 1: formulate query
        Q = self.formulate_query(task, obs_tree)

        # Step 2: web search
        K_web = web_search(Q)

        # Step 3: retrieve narrative memory
        narrative_entries = self.Mn.retrieve(Q, k=3)
        narrative_summaries = [e.summary for e in narrative_entries]

        # Step 4: fuse
        K_fused = self.fuse_knowledge(Q, K_web, narrative_summaries)

        # Step 5: plan subtasks
        subtask_queue = self.plan_subtasks(task, K_fused)

        return Q, subtask_queue

    def replan(self, task: str, current_obs_tree: str, completed: list[dict], failed_subtask: dict, reason: str) -> list[dict]:
        """
        Replan remaining subtasks after a FAIL or PARTIAL_FAIL signal.
        Returns updated subtask queue from the current state onward.
        """
        system = (
            "You are a replanning component for a GUI automation agent. "
            "A subtask has failed. Given the current GUI state and what has been completed, "
            "generate a revised JSON subtask queue to complete the original task. "
            "Output ONLY a JSON array of remaining subtasks."
        )
        completed_names = [s["name"] for s in completed]
        user = (
            f"Original Task: {task}\n"
            f"Completed subtasks: {completed_names}\n"
            f"Failed subtask: {failed_subtask['name']} — Reason: {reason}\n"
            f"Current GUI state:\n{current_obs_tree[:2000]}"
        )
        raw = _gpt(system, user, max_tokens=600)
        raw = re.sub(r"```json|```", "", raw).strip()
        try:
            return json.loads(raw)
        except Exception:
            return [{"name": "Retry", "description": task, "context": ""}]
