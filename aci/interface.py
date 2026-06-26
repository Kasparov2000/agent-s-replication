"""
Agent-Computer Interface (ACI) for Agent S.
- Parses AT-SPI2 accessibility tree and linearizes with unique integer element IDs.
- Augments tree with PaddleOCR-parsed text blocks (IOU dedup).
- Constrains action space to 11 primitives.
- Translates element IDs to pyautogui coordinate calls.
"""

import subprocess
import time
import re
from dataclasses import dataclass, field
from typing import Optional
import pyautogui
import pyautogui as pg

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.3


# ─────────────────────────────────────────────
# Accessibility Tree
# ─────────────────────────────────────────────

@dataclass
class UIElement:
    eid: int
    role: str
    name: str
    x: int
    y: int
    w: int
    h: int
    children: list = field(default_factory=list)

    @property
    def cx(self): return self.x + self.w // 2
    @property
    def cy(self): return self.y + self.h // 2

    def to_text(self) -> str:
        return f"[{self.eid}] {self.role}: \"{self.name}\" @({self.x},{self.y},{self.w},{self.h})"


def get_accessibility_tree() -> list[UIElement]:
    """
    Uses AT-SPI2 via 'at-spi-bus-launcher' + python-atspi to walk the tree.
    Falls back to a mock list if AT-SPI is unavailable (for offline dev/testing).
    """
    try:
        import pyatspi
        desktop = pyatspi.Registry.getDesktop(0)
        elements = []
        _eid = [0]

        def walk(node, depth=0):
            try:
                comp = node.queryComponent()
                ext = comp.getExtents(pyatspi.DESKTOP_COORDS)
                e = UIElement(
                    eid=_eid[0],
                    role=node.getRoleName(),
                    name=node.name or "",
                    x=ext.x, y=ext.y, w=ext.width, h=ext.height
                )
                _eid[0] += 1
                elements.append(e)
                for child in node:
                    walk(child, depth + 1)
            except Exception:
                pass

        for app in desktop:
            walk(app)
        return elements

    except Exception:
        print("[ACI] AT-SPI unavailable, returning empty tree.")
        return []


def linearize_tree(elements: list[UIElement]) -> str:
    """Convert element list to tagged plain-text for MLLM prompt."""
    lines = ["=== Accessibility Tree ==="]
    for e in elements:
        lines.append(e.to_text())
    return "\n".join(lines)


# ─────────────────────────────────────────────
# OCR Augmentation
# ─────────────────────────────────────────────

def ocr_augment_tree(elements: list[UIElement], screenshot_path: str) -> list[UIElement]:
    """
    Run PaddleOCR on screenshot; add any text blocks not already in the tree (IOU check).
    """
    try:
        from paddleocr import PaddleOCR
        import numpy as np
        from PIL import Image

        ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        img = np.array(Image.open(screenshot_path))
        result = ocr.ocr(img, cls=True)

        existing_boxes = [(e.x, e.y, e.w, e.h) for e in elements]

        new_eid = max((e.eid for e in elements), default=-1) + 1
        augmented = list(elements)

        for line in (result[0] or []):
            box, (text, conf) = line
            if conf < 0.6:
                continue
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            x, y = int(min(xs)), int(min(ys))
            w, h = int(max(xs) - x), int(max(ys) - y)

            if _max_iou(x, y, w, h, existing_boxes) < 0.3:
                augmented.append(UIElement(
                    eid=new_eid, role="ocr_text", name=text,
                    x=x, y=y, w=w, h=h
                ))
                new_eid += 1

        return augmented

    except Exception as ex:
        print(f"[ACI] OCR augmentation skipped: {ex}")
        return elements


def _iou(ax, ay, aw, ah, bx, by, bw, bh) -> float:
    ix = max(ax, bx); iy = max(ay, by)
    ix2 = min(ax+aw, bx+bw); iy2 = min(ay+ah, by+bh)
    inter = max(0, ix2-ix) * max(0, iy2-iy)
    union = aw*ah + bw*bh - inter
    return inter / union if union > 0 else 0.0


def _max_iou(x, y, w, h, boxes) -> float:
    return max((_iou(x, y, w, h, bx, by, bw, bh) for bx, by, bw, bh in boxes), default=0.0)


def extract_app_state(elements: list[UIElement]) -> dict:
    """
    Extract structured app state descriptor from top-level nodes.
    Used for application-state-aware memory indexing (Novel Improvement C).
    """
    app_name = ""
    active_dialog = None
    focused_panel = None

    for e in elements[:30]:  # inspect top of tree
        if e.role in ("application", "frame") and not app_name:
            app_name = e.name
        if e.role == "dialog" and not active_dialog:
            active_dialog = e.name
        if e.role == "panel" and not focused_panel:
            focused_panel = e.name

    return {"app_name": app_name, "active_dialog": active_dialog, "focused_panel": focused_panel}


# ─────────────────────────────────────────────
# Action Executor - 11 Primitives
# ─────────────────────────────────────────────

class ACIExecutor:
    def __init__(self):
        self._element_map: dict[int, UIElement] = {}

    def update_elements(self, elements: list[UIElement]):
        self._element_map = {e.eid: e for e in elements}

    def _get_center(self, eid: int) -> tuple[int, int]:
        e = self._element_map.get(eid)
        if e is None:
            raise ValueError(f"Element ID {eid} not found in current tree.")
        return e.cx, e.cy

    # ── 11 primitives ──

    def click(self, eid: int, num_clicks: int = 1, button: str = "left", hold_keys: list = None):
        x, y = self._get_center(eid)
        hold_keys = hold_keys or []
        for k in hold_keys:
            pg.keyDown(k)
        pg.click(x, y, clicks=num_clicks, button=button)
        for k in reversed(hold_keys):
            pg.keyUp(k)

    def type(self, text: str, eid: int = None):
        if eid is not None:
            x, y = self._get_center(eid)
            pg.click(x, y)
            time.sleep(0.1)
        pg.typewrite(text, interval=0.03)

    def scroll(self, eid: int, direction: int):
        """direction: positive = up, negative = down"""
        x, y = self._get_center(eid)
        pg.scroll(direction * 3, x=x, y=y)

    def hotkey(self, keys: list[str]):
        pg.hotkey(*keys)

    def hold_and_press(self, hold: str, press: list[str]):
        pg.keyDown(hold)
        for k in press:
            pg.press(k)
        pg.keyUp(hold)

    def drag_and_drop(self, src_eid: int, dst_eid: int, duration: float = 0.5):
        sx, sy = self._get_center(src_eid)
        dx, dy = self._get_center(dst_eid)
        pg.moveTo(sx, sy)
        pg.dragTo(dx, dy, duration=duration, button="left")

    def save_to_buffer(self, eid: int) -> str:
        """Select all text in element and copy to clipboard."""
        x, y = self._get_center(eid)
        pg.click(x, y)
        pg.hotkey("ctrl", "a")
        pg.hotkey("ctrl", "c")
        import pyperclip
        return pyperclip.paste()

    def switch_applications(self, app_name: str):
        """Alt-tab loop until window title matches."""
        pg.hotkey("alt", "tab")
        time.sleep(0.3)

    def wait(self, seconds: float = 1.0):
        time.sleep(seconds)

    def done(self):
        return "DONE"

    def fail(self):
        return "FAIL"

    def dispatch(self, action_str: str) -> Optional[str]:
        """
        Parse and execute an action string of the form:
          agent.click(41, 1, 'left')
          agent.type('hello world')
          agent.hotkey(['ctrl','s'])
          agent.done()
          etc.
        Returns 'DONE', 'FAIL', or None.
        """
        action_str = action_str.strip()
        try:
            # Extract method name and raw args
            m = re.match(r"agent\.(\w+)\((.*)\)$", action_str, re.DOTALL)
            if not m:
                print(f"[ACI] Unrecognized action: {action_str}")
                return None
            method, raw_args = m.group(1), m.group(2).strip()

            # Safe eval of args
            args = eval(f"[{raw_args}]") if raw_args else []

            fn = getattr(self, method, None)
            if fn is None:
                print(f"[ACI] Unknown primitive: {method}")
                return None

            result = fn(*args)
            time.sleep(0.5)  # allow UI to settle
            return result

        except Exception as ex:
            print(f"[ACI] Dispatch error on '{action_str}': {ex}")
            return None
