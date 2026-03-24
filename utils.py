"""
Shared helpers for Blender extension UI and logic.
"""

from __future__ import annotations

import textwrap


def selected_keyframes_for_object(obj) -> list[int]:
    animation_data = getattr(obj, "animation_data", None)
    action = getattr(animation_data, "action", None) if animation_data else None
    if action is None:
        return []

    frames: set[int] = set()
    for fcurve in action.fcurves:
        for keyframe in fcurve.keyframe_points:
            if keyframe.select_control_point:
                frames.add(int(round(keyframe.co.x)))
    return sorted(frames)


def wrap_text_to_panel(text: str, context, *, min_chars: int = 8, full_width: bool = False) -> str:
    """Wrap a string into multiple lines based on the current UI panel width.

    UI usage pattern:
        wrapped = wrap_text_to_panel("Long message", context, full_width=True)
        for line in (wrapped.splitlines() or [""]):
            layout.label(text=line)
    """
    try:
        width = getattr(context.region, "width", 300) or 300
        prefs = getattr(context, "preferences", None)
        view = getattr(prefs, "view", None) if prefs else None
        scale = getattr(view, "ui_scale", 1.0) if view else 1.0

        # Reserve pixels for icons/indents. full_width assumes a single-column layout.
        reserved = 240 if not full_width else 110
        available = max(50, width - reserved)

        px_per_char = (13.5 if not full_width else 9.5) * max(scale, 0.5)
        max_chars = max(min_chars, int(available / px_per_char) - 8)
    except Exception:
        max_chars = min_chars

    # Cap to keep wrap reasonable even on very wide panels.
    max_cap = 75 if not full_width else 260
    max_chars = max(min_chars, min(max_cap, max_chars))

    return textwrap.fill(
        text or "",
        width=max_chars,
        break_long_words=False,
        break_on_hyphens=False,
        replace_whitespace=False,
        expand_tabs=False,
    )
