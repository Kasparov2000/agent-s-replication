import os
import base64
import numpy as np

PROVIDER = os.environ.get("LLM_PROVIDER", "openai").lower()
EMBED_DIM = 1536

from openai import OpenAI
_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
CHAT_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"

def chat(system: str, user: str, max_tokens: int = 1024) -> str:
    resp = _client.chat.completions.create(
        model=CHAT_MODEL,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    )
    return resp.choices[0].message.content.strip()

def vision(image_b64: str, prompt: str, max_tokens: int = 80) -> str:
    resp = _client.chat.completions.create(
        model=CHAT_MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            {"type": "text", "text": prompt}
        ]}]
    )
    return resp.choices[0].message.content.strip()

def embed(text: str) -> np.ndarray:
    r = _client.embeddings.create(model=EMBED_MODEL, input=text)
    v = np.array(r.data[0].embedding, dtype=np.float32)
    return v / np.linalg.norm(v)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))
