"""
Memory system for Agent S.
Narrative Memory (Mn): full-task experience summaries, indexed by query Q.
Episodic Memory (Me): step-by-step subtask trajectories, indexed by (Q, subtask, context).
Both use FAISS + text-embedding-3-small for retrieval.
"""

import json
import os
import pickle
import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional
from openai import OpenAI

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("[WARN] faiss not installed. Memory retrieval disabled.")

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))


def embed(text: str) -> np.ndarray:
    response = client.embeddings.create(model=EMBED_MODEL, input=text)
    vec = np.array(response.data[0].embedding, dtype=np.float32)
    return vec / np.linalg.norm(vec)  # normalize for cosine sim


@dataclass
class NarrativeEntry:
    query: str
    summary: str       # high-level strategy, key decisions, outcome
    task: str
    success: bool


@dataclass
class EpisodicEntry:
    key: str           # concatenation of task + subtask + context + app_state
    summary: str       # step-by-step action sequence in human-readable form
    subtask: str
    app_name: str
    active_dialog: Optional[str]
    focused_panel: Optional[str]


class NarrativeMemory:
    """Stores full-task summaries. Used by Manager for planning."""

    def __init__(self, path: str = "narrative_memory.pkl"):
        self.path = path
        self.entries: list[NarrativeEntry] = []
        self.vectors: list[np.ndarray] = []
        self.index = None
        self._load()

    def _build_index(self):
        if not FAISS_AVAILABLE or not self.vectors:
            return
        mat = np.stack(self.vectors)
        self.index = faiss.IndexFlatIP(EMBED_DIM)  # inner product = cosine (normalized)
        self.index.add(mat)

    def add(self, entry: NarrativeEntry):
        vec = embed(entry.query)
        self.entries.append(entry)
        self.vectors.append(vec)
        self._build_index()
        self._save()

    def retrieve(self, query: str, k: int = 3) -> list[NarrativeEntry]:
        if not self.index or not self.entries:
            return []
        q_vec = embed(query).reshape(1, -1)
        k = min(k, len(self.entries))
        scores, indices = self.index.search(q_vec, k)
        return [self.entries[i] for i in indices[0] if i >= 0]

    def _save(self):
        with open(self.path, "wb") as f:
            pickle.dump({"entries": self.entries, "vectors": self.vectors}, f)

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, "rb") as f:
                data = pickle.load(f)
            self.entries = data["entries"]
            self.vectors = data["vectors"]
            self._build_index()


class EpisodicMemory:
    """Stores step-by-step subtask trajectories. Used by Workers."""

    def __init__(self, path: str = "episodic_memory.pkl"):
        self.path = path
        self.entries: list[EpisodicEntry] = []
        self.vectors: list[np.ndarray] = []
        self.index = None
        self._load()

    def _build_index(self):
        if not FAISS_AVAILABLE or not self.vectors:
            return
        mat = np.stack(self.vectors)
        self.index = faiss.IndexFlatIP(EMBED_DIM)
        self.index.add(mat)

    def _make_key(self, task: str, subtask: str, context: str, app_state: dict) -> str:
        """Application-state-aware key (Novel Improvement C)."""
        state_str = f"{app_state.get('app_name','')}|{app_state.get('active_dialog','none')}|{app_state.get('focused_panel','none')}"
        return f"{task} | {subtask} | {context} | {state_str}"

    def add(self, task: str, subtask: str, context: str, app_state: dict, summary: str):
        key = self._make_key(task, subtask, context, app_state)
        entry = EpisodicEntry(
            key=key,
            summary=summary,
            subtask=subtask,
            app_name=app_state.get("app_name", ""),
            active_dialog=app_state.get("active_dialog"),
            focused_panel=app_state.get("focused_panel"),
        )
        vec = embed(key)
        self.entries.append(entry)
        self.vectors.append(vec)
        self._build_index()
        self._save()

    def retrieve(self, task: str, subtask: str, context: str, app_state: dict, k: int = 3) -> list[EpisodicEntry]:
        """Two-stage retrieval: filter by app_name, then rank by embedding similarity."""
        if not self.index or not self.entries:
            return []

        # Stage 1: filter by exact app_name match
        app_name = app_state.get("app_name", "")
        candidate_indices = [
            i for i, e in enumerate(self.entries)
            if e.app_name == app_name or not app_name
        ]
        if not candidate_indices:
            candidate_indices = list(range(len(self.entries)))

        # Stage 2: rank by embedding similarity
        query_key = self._make_key(task, subtask, context, app_state)
        q_vec = embed(query_key).reshape(1, -1)

        candidate_vecs = np.stack([self.vectors[i] for i in candidate_indices])
        scores = (candidate_vecs @ q_vec.T).flatten()
        top_k = min(k, len(candidate_indices))
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [self.entries[candidate_indices[i]] for i in top_idx]

    def _save(self):
        with open(self.path, "wb") as f:
            pickle.dump({"entries": self.entries, "vectors": self.vectors}, f)

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, "rb") as f:
                data = pickle.load(f)
            self.entries = data["entries"]
            self.vectors = data["vectors"]
            self._build_index()
