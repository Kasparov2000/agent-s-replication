"""
Self-Evaluator for Agent S.
- After successful subtask: generates episodic experience summary → stores in Me.
- After full task: generates narrative experience summary → stores in Mn.
- Operates without ground-truth labels — pure MLLM self-assessment.
"""

import os
from openai import OpenAI
from memory.memory_store import NarrativeMemory, EpisodicMemory, NarrativeEntry

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
MODEL = "gpt-4o"


def _gpt(system: str, user: str, max_tokens: int = 512) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    )
    return resp.choices[0].message.content.strip()


class SelfEvaluator:
    def __init__(self, narrative_memory: NarrativeMemory, episodic_memory: EpisodicMemory):
        self.Mn = narrative_memory
        self.Me = episodic_memory

    def summarize_subtask(
        self,
        task: str,
        subtask: dict,
        step_log: list,
        app_state: dict,
    ) -> str:
        """
        Generate and store an episodic experience summary for a completed subtask.
        Returns the summary string.
        """
        step_text = "\n".join(
            [f"Step {r.step}: {r.action} (status={r.status})" for r in step_log]
        )
        system = (
            "You are summarizing a completed subtask trajectory for future reuse by a GUI agent. "
            "Produce a concise, human-readable step-by-step procedure that another agent could follow. "
            "Include specific element IDs and action types where relevant. "
            "Format: 'To <subtask>: 1. <action> 2. <action> ...'"
        )
        user = (
            f"Task: {task}\n"
            f"Subtask: {subtask['name']} — {subtask['description']}\n"
            f"Application: {app_state.get('app_name', 'unknown')}\n\n"
            f"Executed steps:\n{step_text}"
        )
        summary = _gpt(system, user, max_tokens=300)

        # Store in episodic memory
        self.Me.add(
            task=task,
            subtask=subtask["name"],
            context=subtask.get("context", ""),
            app_state=app_state,
            summary=summary,
        )
        return summary

    def summarize_task(
        self,
        task: str,
        query: str,
        subtask_summaries: list[str],
        success: bool,
        final_obs_tree: str,
    ) -> str:
        """
        Generate and store a narrative experience summary for the full task.
        Returns the summary string.
        """
        subs_text = "\n".join([f"- {s}" for s in subtask_summaries])
        outcome = "SUCCESS" if success else "FAILURE (step limit or unrecoverable FAIL)"

        system = (
            "You are summarizing a completed task trajectory for a GUI automation agent's long-term memory. "
            "Include: overall strategy, key decision points, what worked, what failed, and the outcome. "
            "Be concise but informative. Another agent should be able to learn from this summary."
        )
        user = (
            f"Task: {task}\n"
            f"Outcome: {outcome}\n\n"
            f"Subtask summaries:\n{subs_text}\n\n"
            f"Final GUI state snippet:\n{final_obs_tree[:500]}"
        )
        summary = _gpt(system, user, max_tokens=400)

        # Store in narrative memory
        entry = NarrativeEntry(
            query=query,
            summary=summary,
            task=task,
            success=success,
        )
        self.Mn.add(entry)
        return summary
