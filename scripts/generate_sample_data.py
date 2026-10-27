#!/usr/bin/env python3
"""
Generate Synthetic Activity Clips — Enhanced Visual Distinctiveness
====================================================================
Each class gets a UNIQUE colour scheme (background + clothing) so the
frozen MobileNetV3 backbone can separate them without fine motion cues.

Class → visual identity:
  SittingNormally  → deep blue bg,  red shirt,   seated still
  RaisingHand      → dark green bg, yellow shirt, arm sweeps up (large arc)
  ThumbsUp         → dark red bg,   cyan shirt,   standing, thumb visible
  ScratchingHead   → purple bg,     orange shirt, hand on head circles
  LookingAway      → teal bg,       white shirt,  head profile + turn arrow

8 clips per class, each with a slightly different variant offset
so the dataset has some within-class variation.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

ACTIONS = [
    "SittingNormally",
    "RaisingHand",
    "ThumbsUp",
    "ScratchingHead",
    "LookingAway",
]

# Unique (background BGR, shirt BGR) per class
CLASS_PALETTE = {
    "SittingNormally": ((80, 40, 10),   (30,  30, 200)),   # blue bg, red shirt
    "RaisingHand":     ((10, 70, 10),   (20, 220, 220)),   # green bg, yellow shirt
    "ThumbsUp":        ((20, 10, 80),   (200, 200,  20)),  # red bg, cyan shirt
    "ScratchingHead":  ((60, 10, 60),   (20, 130, 230)),   # purple bg, orange shirt
    "LookingAway":     ((60, 80, 10),   (230, 230, 230)),  # teal bg, white shirt
}

DATASET_ROOT = Path(__file__).resolve().parent.parent / "data" / "ucf101"


def draw_scene(img: np.ndarray, t: float, action: str, variant: int = 0) -> None:
    """Render one frame for `action` at phase t∈[0,1)."""
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2

    bg, shirt = CLASS_PALETTE[action]

    # ── Background (solid class colour + subtle noise) ─────────────────────
    img[:] = np.array(bg, dtype=np.uint8)
    noise = np.random.randint(0, 12, (h, w, 3), dtype=np.uint8)
    cv2.add(img, noise, img)

    # Slight variant brightness offset so clips within a class differ
    bright = int(variant * 6)
    img[:] = np.clip(img.astype(np.int16) + bright, 0, 255).astype(np.uint8)

    # Floor line
    cv2.line(img, (10, h - 20), (w - 10, h - 20), (100, 100, 100), 2)

    skin   = (150, 190, 220)   # same across all (light skin BGR)
    joints = (240, 220, 180)
    TH = 3                      # limb thickness

    s = math.sin(t * 2 * math.pi)
    c = math.cos(t * 2 * math.pi)

    # ── Chair helper ────────────────────────────────────────────────────────
    def chair(seat_y: int) -> None:
        cv2.rectangle(img, (cx-38, seat_y), (cx+38, seat_y+10), (40,30,20), -1)
        cv2.rectangle(img, (cx-38, seat_y-52), (cx-30, seat_y), (40,30,20), -1)
        cv2.rectangle(img, (cx+30, seat_y-52), (cx+38, seat_y), (40,30,20), -1)
        cv2.line(img, (cx-38, seat_y-52), (cx+38, seat_y-52), (55,42,28), 2)

    # ── SittingNormally ─────────────────────────────────────────────────────
    if action == "SittingNormally":
        seat_y = cy + 25
        chair(seat_y)
        sway = int(3 * s)

        # Head
        cv2.circle(img, (cx+sway, seat_y-72), 18, skin, -1)
        # Eyes (signal "looking forward")
        cv2.circle(img, (cx+sway-6, seat_y-76), 3, (20,20,20), -1)
        cv2.circle(img, (cx+sway+6, seat_y-76), 3, (20,20,20), -1)
        # Torso
        cv2.line(img, (cx+sway, seat_y-54), (cx, seat_y+8), shirt, TH+2)
        # Arms: both resting on thighs
        cv2.line(img, (cx-4, seat_y-38), (cx-26, seat_y+2), skin, TH)
        cv2.line(img, (cx-26, seat_y+2), (cx-28, seat_y+16), skin, TH)
        cv2.line(img, (cx+4, seat_y-38), (cx+26, seat_y+2), skin, TH)
        cv2.line(img, (cx+26, seat_y+2), (cx+28, seat_y+16), skin, TH)
        # Legs: seated
        cv2.line(img, (cx, seat_y+8), (cx-22, seat_y+40), skin, TH)
        cv2.line(img, (cx-22, seat_y+40), (cx-24, h-22), skin, TH)
        cv2.line(img, (cx, seat_y+8), (cx+22, seat_y+40), skin, TH)
        cv2.line(img, (cx+22, seat_y+40), (cx+24, h-22), skin, TH)

    # ── RaisingHand ─────────────────────────────────────────────────────────
    elif action == "RaisingHand":
        seat_y = cy + 25
        chair(seat_y)

        # Head
        cv2.circle(img, (cx, seat_y-72), 18, skin, -1)
        # Torso
        cv2.line(img, (cx, seat_y-54), (cx, seat_y+8), shirt, TH+2)

        # RIGHT arm: big arc from hip → overhead (t drives full sweep)
        angle = t * math.pi               # 0 → π  (down to up)
        shoulder = (cx+6, seat_y-42)
        elbow = (
            int(shoulder[0] + 30 * math.cos(angle - math.pi/2)),
            int(shoulder[1] - 30 * math.sin(angle - math.pi/2)),
        )
        hand = (
            int(elbow[0] + 30 * math.cos(angle - math.pi/2)),
            int(elbow[1] - 30 * math.sin(angle - math.pi/2)),
        )
        cv2.line(img, shoulder, elbow, skin, TH+1)
        cv2.line(img, elbow, hand, skin, TH+1)
        cv2.circle(img, hand, 7, joints, -1)   # bright hand

        # LEFT arm resting
        cv2.line(img, (cx-4, seat_y-38), (cx-26, seat_y+6), skin, TH)
        cv2.line(img, (cx-26, seat_y+6), (cx-28, seat_y+18), skin, TH)

        # Legs
        cv2.line(img, (cx, seat_y+8), (cx-22, seat_y+40), skin, TH)
        cv2.line(img, (cx-22, seat_y+40), (cx-24, h-22), skin, TH)
        cv2.line(img, (cx, seat_y+8), (cx+22, seat_y+40), skin, TH)
        cv2.line(img, (cx+22, seat_y+40), (cx+24, h-22), skin, TH)

    # ── ThumbsUp ────────────────────────────────────────────────────────────
    elif action == "ThumbsUp":
        stand_y = cy + 35   # hip
        # Head (centred, facing camera)
        cv2.circle(img, (cx, stand_y-90), 20, skin, -1)
        cv2.circle(img, (cx-7, stand_y-94), 3, (20,20,20), -1)
        cv2.circle(img, (cx+7, stand_y-94), 3, (20,20,20), -1)
        # Torso (tall, standing)
        cv2.line(img, (cx, stand_y-70), (cx, stand_y), shirt, TH+2)

        # Right arm: bent at elbow, forearm vertical, fist up, LARGE thumb
        shoulder = (cx+8, stand_y-55)
        elbow    = (cx+45, stand_y-32 + int(4*s))
        wrist    = (cx+45, stand_y-68 + int(5*s))
        cv2.line(img, shoulder, elbow, skin, TH+1)
        cv2.line(img, elbow, wrist, skin, TH+1)
        # Fist
        cv2.circle(img, wrist, 10, skin, -1)
        # Prominent THUMB — bright yellow, big
        thumb_base = (wrist[0]-3, wrist[1]-10)
        thumb_tip  = (wrist[0]-7, wrist[1]-28)
        cv2.line(img, thumb_base, thumb_tip, (0, 230, 230), 6)
        cv2.circle(img, thumb_tip, 7, (0, 255, 255), -1)

        # Left arm hanging
        cv2.line(img, (cx-8, stand_y-55), (cx-18, stand_y-18), skin, TH)
        cv2.line(img, (cx-18, stand_y-18), (cx-20, stand_y+2), skin, TH)

        # Legs standing
        cv2.line(img, (cx, stand_y), (cx-18, h-22), skin, TH)
        cv2.line(img, (cx, stand_y), (cx+18, h-22), skin, TH)

    # ── ScratchingHead ──────────────────────────────────────────────────────
    elif action == "ScratchingHead":
        seat_y = cy + 25
        chair(seat_y)

        # Head (stays centred — hand moves around it)
        cv2.circle(img, (cx, seat_y-72), 18, skin, -1)
        cv2.circle(img, (cx-6, seat_y-76), 3, (20,20,20), -1)
        cv2.circle(img, (cx+6, seat_y-76), 3, (20,20,20), -1)
        # Torso
        cv2.line(img, (cx, seat_y-54), (cx, seat_y+8), shirt, TH+2)

        # RIGHT arm: hand orbits around head
        radius = 24
        scratch_x = cx + int(radius * c)
        scratch_y = seat_y - 72 + int(radius * 0.5 * s)
        shoulder = (cx+6, seat_y-42)
        elbow    = (cx+28, seat_y-55)
        cv2.line(img, shoulder, elbow, skin, TH+1)
        cv2.line(img, elbow, (scratch_x, scratch_y), skin, TH+1)
        # Hand circle (bright) on / around head
        cv2.circle(img, (scratch_x, scratch_y), 8, joints, -1)
        cv2.circle(img, (scratch_x, scratch_y), 8, (0, 180, 255), 2)

        # Left arm resting
        cv2.line(img, (cx-4, seat_y-38), (cx-26, seat_y+6), skin, TH)
        cv2.line(img, (cx-26, seat_y+6), (cx-28, seat_y+18), skin, TH)

        # Legs
        cv2.line(img, (cx, seat_y+8), (cx-22, seat_y+40), skin, TH)
        cv2.line(img, (cx-22, seat_y+40), (cx-24, h-22), skin, TH)
        cv2.line(img, (cx, seat_y+8), (cx+22, seat_y+40), skin, TH)
        cv2.line(img, (cx+22, seat_y+40), (cx+24, h-22), skin, TH)

    # ── LookingAway ─────────────────────────────────────────────────────────
    elif action == "LookingAway":
        seat_y = cy + 25
        chair(seat_y)

        # Head turns: profile ellipse narrows when facing side
        turn = math.sin(t * math.pi)          # -1..1
        head_cx = cx + int(16 * turn)
        rx = max(6, int(18 * (1.0 - 0.55 * abs(turn))))
        # Head ellipse
        cv2.ellipse(img, (head_cx, seat_y-72), (rx, 18), 0, 0, 360, skin, -1)
        cv2.ellipse(img, (head_cx, seat_y-72), (rx, 18), 0, 0, 360, (200,180,160), 2)

        # LARGE arrow showing gaze direction (very distinct)
        arrow_dx = int(45 * turn)
        arrow_start = (head_cx, seat_y - 72)
        arrow_end   = (head_cx + arrow_dx, seat_y - 72)
        if arrow_dx != 0:
            cv2.arrowedLine(img, arrow_start, arrow_end, (0, 255, 200), 3, tipLength=0.4)

        # Torso
        cv2.line(img, (cx, seat_y-54), (cx, seat_y+8), shirt, TH+2)
        # Arms at rest
        cv2.line(img, (cx-4, seat_y-38), (cx-26, seat_y+6), skin, TH)
        cv2.line(img, (cx-26, seat_y+6), (cx-28, seat_y+18), skin, TH)
        cv2.line(img, (cx+4, seat_y-38), (cx+26, seat_y+6), skin, TH)
        cv2.line(img, (cx+26, seat_y+6), (cx+28, seat_y+18), skin, TH)
        # Legs
        cv2.line(img, (cx, seat_y+8), (cx-22, seat_y+40), skin, TH)
        cv2.line(img, (cx-22, seat_y+40), (cx-24, h-22), skin, TH)
        cv2.line(img, (cx, seat_y+8), (cx+22, seat_y+40), skin, TH)
        cv2.line(img, (cx+22, seat_y+40), (cx+24, h-22), skin, TH)


def generate_video(path: Path, action: str, num_frames: int = 32,
                   fps: int = 25, variant: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, fps, (256, 256))
    for i in range(num_frames):
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        draw_scene(img, i / float(num_frames), action, variant)
        out.write(img)
    out.release()


def main() -> None:
    clips_per_class = 8
    print(f"Generating {len(ACTIONS)*clips_per_class} high-contrast synthetic clips …")
    print()
    for action in ACTIONS:
        cls_dir = DATASET_ROOT / action
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(1, clips_per_class + 1):
            vpath = cls_dir / f"v_{action}_g{i:02d}_c01.mp4"
            generate_video(vpath, action, num_frames=32, fps=25, variant=i - 1)
            print(f"  {vpath.relative_to(DATASET_ROOT.parent)}")
    print()
    print(f"Done. Classes ({len(ACTIONS)}): {ACTIONS}")
    print()
    print("NOTE: synthetic clips only — replace with real videos for production accuracy.")
    print("Folder structure expected by the loader:")
    for a in ACTIONS:
        print(f"  {DATASET_ROOT / a}/")


if __name__ == "__main__":
    main()
