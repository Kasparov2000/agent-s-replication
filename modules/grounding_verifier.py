"""
Grounding Verifier (Novel Improvement A).
Before executing click/type/drag, crops the screenshot to the proposed element's
bounding box, describes it with GPT-4o-mini, and compares against the action's
semantic intent via cosine similarity of embeddings.
If similarity < threshold, rejects the action and returns a re-grounding signal.
"""

import base64
import os
import numpy as np
from openai import OpenAI
from PIL import Image
import io

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
THRESHOLD = 0.75
EMBED_MODEL = "text-embedding-3-small"
VISION_MODEL = "gpt-4o-mini"

VERIFIED_PRIMITIVES = {"click", "type", "drag_and_drop"}


def _embed(text: str) -> np.ndarray:
    r = client.embeddings.create(model=EMBED_MODEL, input=text)
    v = np.array(r.data[0].embedding, dtype=np.float32)
    return v / np.linalg.norm(v)


def _crop_to_base64(screenshot_path: str, x: int, y: int, w: int, h: int) -> str:
    img = Image.open(screenshot_path)
    # pad slightly for context
    pad = 10
    box = (max(0, x-pad), max(0, y-pad), x+w+pad, y+h+pad)
    crop = img.crop(box)
    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _describe_crop(b64_img: str) -> str:
    """Ask GPT-4o-mini to describe what's visible in the cropped region."""
    response = client.chat.completions.create(
        model=VISION_MODEL,
        max_tokens=80,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}},
                {"type": "text", "text": "In one sentence, describe what UI element is shown in this image."}
            ]
        }]
    )
    return response.choices[0].message.content.strip()


def verify_grounding(
    action_str: str,
    semantic_intent: str,
    screenshot_path: str,
    element_map: dict,
) -> tuple[bool, str]:
    """
    Returns (ok, feedback).
    - ok=True: action passes; proceed to execute.
    - ok=False: action rejected; feedback contains visual description for re-grounding.
    """
    import re

    # Only verify grounded primitives
    m = re.match(r"agent\.(\w+)\((\d+)", action_str)
    if not m or m.group(1) not in VERIFIED_PRIMITIVES:
        return True, ""

    eid = int(m.group(2))
    elem = element_map.get(eid)
    if elem is None:
        return False, f"Element ID {eid} not found in current tree."

    try:
        b64 = _crop_to_base64(screenshot_path, elem.x, elem.y, elem.w, elem.h)
        visual_desc = _describe_crop(b64)

        # Compare visual description vs semantic intent
        v_intent = _embed(semantic_intent)
        v_visual = _embed(visual_desc)
        sim = float(np.dot(v_intent, v_visual))

        if sim >= THRESHOLD:
            return True, ""
        else:
            feedback = (
                f"Grounding mismatch (similarity={sim:.2f}). "
                f"Element {eid} appears to be: '{visual_desc}'. "
                f"Intended target was: '{semantic_intent}'. "
                f"Please select a different element ID."
            )
            return False, feedback

    except Exception as ex:
        # If verifier fails, allow action through (fail-open)
        print(f"[GV] Verifier error: {ex} — allowing action.")
        return True, ""
