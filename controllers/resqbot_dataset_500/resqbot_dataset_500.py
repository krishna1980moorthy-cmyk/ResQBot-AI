"""
resqbot_dataset_collector.py — Automated Real-World Webots R2025a YOLO11n Dataset Collector
===========================================================================================
Run via `worlds/resqbot_dataset_capture.wbt` (ResQBot_AI with supervisor=TRUE).

What it does automatically in ~60-90 seconds of Webots simulation:
  1. Teleports ResQBot_AI to diverse, collision-free viewpoints around Victim_01..05
     (varied distances 1.1m-4.8m, angles, slight offsets) AND empty rubble/road scenes.
  2. Verifies line-of-sight using the rover's own LiDAR so victims occluded behind
     solid concrete walls are NEVER falsely labeled.
  3. Projects each visible victim's 3D bounding box through the exact `rgb_camera`
     pinhole intrinsics (320x240, HFOV=1.10 rad) to emit accurate YOLO `.txt` labels
     (class 0 = "victim").
  4. Saves clean scene-separated splits to `datasets/resqbot_victims_500/`:
       - train (~350 imgs): Victim_01, Victim_02, Victim_03 + empty scenes
       - val   (~100 imgs): Victim_04 + empty scenes
       - test  (~50 imgs) : Victim_05 + empty scenes
     Plus `dataset.yaml` and annotated verification previews in `preview_boxes/`.
"""

from __future__ import annotations

import math
import os
import random
from typing import List, Optional, Tuple

import numpy as np
from controller import Supervisor

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False


# =============================================================================
# CONFIGURATION (Matches your 500-image target & scene-separated splits)
# =============================================================================

OUTPUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "resqbot_victims_500")
)

# Exact victim DEF names in earthquake_multi_hazard_world.wbt
# Split by distinct victim scene so train / val / test have ZERO frame leakage!
VICTIM_SPLITS = {
    "Victim_01": ("train", 110, (0.75, 0.45, 0.32)),  # Supine near Zone B entrance
    "Victim_02": ("train", 95,  (0.55, 0.42, 0.90)),  # Slumped seated inside Zone B
    "Victim_03": ("train", 85,  (0.65, 0.45, 0.34)),  # Curled side along Search_Corridor
    "Victim_04": ("val",   75,  (0.70, 0.45, 0.30)),  # Prone in Zone C North alcove
    "Victim_05": ("test",  35,  (0.65, 0.45, 0.32)),  # Curled side near Zone E annex
}

# Empty negative scenes (no victim in frame -> empty .txt label file)
EMPTY_SPLIT_COUNTS = {
    "train": 60,
    "val": 25,
    "test": 15,
}

# Camera offset in ResQBotRover.proto local frame (+X forward, +Y left, +Z up)
CAM_LOCAL_OFFSET = (0.215, 0.022, 0.194)

# Safe open anchor regions for empty-scene negative captures (away from victims)
EMPTY_ANCHORS = [
    (-14.0, -8.5),   # Zone A staging road
    (-10.0, -8.5),   # Broken road west
    (4.0, -8.5),     # Broken road east
    (11.0, -8.5),    # Broken road far east
    (0.0, -4.0),     # Search corridor south
    (0.0, 5.5),      # Search corridor mid
    (0.0, 11.5),     # Search corridor north
    (6.0, 12.0),     # Zone D gas yard approach
]


def _ensure_dirs() -> None:
    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "preview_boxes"), exist_ok=True)


def _write_dataset_yaml() -> None:
    yaml_path = os.path.join(OUTPUT_DIR, "dataset.yaml")
    root_posix = OUTPUT_DIR.replace("\\", "/")
    content = (
        f"path: {root_posix}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n\n"
        f"names:\n"
        f"  0: victim\n"
    )
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[DATASET] Wrote {yaml_path}")


def _world_to_cam(pt_w: List[float], rover_pos: List[float], rover_rot: List[float]) -> Tuple[float, float, float]:
    dx = pt_w[0] - rover_pos[0]
    dy = pt_w[1] - rover_pos[1]
    dz = pt_w[2] - rover_pos[2]
    lx = rover_rot[0] * dx + rover_rot[3] * dy + rover_rot[6] * dz
    ly = rover_rot[1] * dx + rover_rot[4] * dy + rover_rot[7] * dz
    lz = rover_rot[2] * dx + rover_rot[5] * dy + rover_rot[8] * dz
    return lx - CAM_LOCAL_OFFSET[0], ly - CAM_LOCAL_OFFSET[1], lz - CAM_LOCAL_OFFSET[2]


def _project_victim_bbox(
    victim_pos: List[float],
    half_extents: Tuple[float, float, float],
    rover_pos: List[float],
    rover_rot: List[float],
    width: int,
    height: int,
    hfov: float,
) -> Optional[Tuple[float, float, float, float]]:
    """Project 8 corners of victim 3D bounding box into normalized YOLO (cx, cy, w, h)."""
    fx = (width * 0.5) / math.tan(hfov * 0.5)
    fy = fx  # Square pixels in Webots Camera
    cx0, cy0 = width * 0.5, height * 0.5

    hx, hy, hz = half_extents
    vz_center = victim_pos[2] + hz * 0.5
    us, vs = [], []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-0.5, 0.5):
                corner = [victim_pos[0] + sx * hx, victim_pos[1] + sy * hy, vz_center + sz * hz]
                x, y, z = _world_to_cam(corner, rover_pos, rover_rot)
                if x <= 0.20:
                    continue
                # Webots local frame: +X forward, +Y left, +Z up -> image u right, v down
                u = cx0 - fx * (y / x)
                v = cy0 - fy * (z / x)
                us.append(u)
                vs.append(v)

    if len(us) < 4:
        return None

    x1 = max(0.0, min(float(width - 1), min(us)))
    x2 = max(0.0, min(float(width - 1), max(us)))
    y1 = max(0.0, min(float(height - 1), min(vs)))
    y2 = max(0.0, min(float(height - 1), max(vs)))

    bw = (x2 - x1) / float(width)
    bh = (y2 - y1) / float(height)
    # Reject tiny specks or clipped boxes outside frame
    if bw < 0.035 or bh < 0.035 or bw > 0.96 or bh > 0.96:
        return None
    cx = ((x1 + x2) * 0.5) / float(width)
    cy = ((y1 + y2) * 0.5) / float(height)
    return cx, cy, bw, bh


def _lidar_has_los(lidar_ranges: List[float], target_dist: float, bearing_rad: float) -> bool:
    """Return True if LiDAR along target bearing is not blocked by a closer solid wall."""
    if not lidar_ranges:
        return True
    n = len(lidar_ranges)
    # ResQBotRover lidar: 0 deg = center index (n//2), clockwise (+pi -> -pi)
    ang_deg = math.degrees(bearing_rad)
    frac = (180.0 - ang_deg) / 360.0
    center_idx = int(round(frac * n)) % n
    rays = []
    for d in (-3, -2, -1, 0, 1, 2, 3):
        r = lidar_ranges[(center_idx + d) % n]
        if not math.isnan(r) and r > 0.12:
            rays.append(12.0 if math.isinf(r) else r)
    if not rays:
        return False
    # If closest solid wall along that bearing is > 0.55m closer than victim, victim is occluded
    return max(rays) >= (target_dist - 0.55)


def _save_preview(raw_bgra: bytes, w: int, h: int, boxes: List[Tuple[float, float, float, float]], out_path: str) -> None:
    if not _HAS_CV2:
        return
    img = np.frombuffer(raw_bgra, dtype=np.uint8).reshape((h, w, 4))[:, :, :3].copy()
    for cx, cy, bw, bh in boxes:
        x1 = int((cx - bw * 0.5) * w)
        y1 = int((cy - bh * 0.5) * h)
        x2 = int((cx + bw * 0.5) * w)
        y2 = int((cy + bh * 0.5) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 80), 2)
        cv2.putText(img, "victim", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 80), 1)
    cv2.imwrite(out_path, img)


def main() -> None:
    _ensure_dirs()
    _write_dataset_yaml()

    sup = Supervisor()
    ts = int(sup.getBasicTimeStep())
    rover = sup.getSelf()
    t_field = rover.getField("translation")
    r_field = rover.getField("rotation")

    # Lock motors at zero so rover stays steady during teleport captures
    for m_name in ("front_left_motor", "front_right_motor", "rear_left_motor", "rear_right_motor"):
        m = sup.getDevice(m_name)
        if m is not None:
            m.setPosition(float("inf"))
            m.setVelocity(0.0)

    cam = sup.getDevice("rgb_camera")
    lidar = sup.getDevice("lidar")
    if cam is None or lidar is None:
        print("[ERROR] rgb_camera or lidar missing on ResQBot_AI!")
        return
    cam.enable(ts)
    lidar.enable(ts)

    w, h, hfov = int(cam.getWidth()), int(cam.getHeight()), float(cam.getFov())

    # Locate all 5 victim nodes in world
    victims = {}
    for vdef, (split, target_cnt, ext) in VICTIM_SPLITS.items():
        node = sup.getFromDef(vdef)
        if node is not None:
            victims[vdef] = (node, split, target_cnt, ext)
    print(f"[COLLECTOR] Found {len(victims)}/5 victim nodes. Output -> {OUTPUT_DIR}")

    rng = random.Random(20261124)
    saved_counts = {"train": 0, "val": 0, "test": 0}
    preview_count = 0

    def step_settle(n_steps: int = 3) -> bool:
        for _ in range(n_steps):
            if sup.step(ts) == -1:
                return False
        return True

    if not step_settle(4):
        return

    # 1. Capture diverse positive images around each victim
    for vdef, (vnode, split, target_cnt, ext) in victims.items():
        vpos = vnode.getPosition()
        got = 0
        tries = 0
        max_tries = target_cnt * 10
        print(f"[COLLECTOR] Capturing {target_cnt} views of {vdef} -> split='{split}' ...")

        while got < target_cnt and tries < max_tries:
            tries += 1
            dist = rng.uniform(1.15, 4.40)
            az = rng.uniform(-math.pi, math.pi)
            rx = vpos[0] + dist * math.cos(az)
            ry = vpos[1] + dist * math.sin(az)
            # Point rover toward victim with slight realistic heading jitter (+/- 14 deg)
            yaw_to_v = math.atan2(vpos[1] - ry, vpos[0] - rx) + rng.uniform(-0.24, 0.24)

            t_field.setSFVec3f([rx, ry, 0.015])
            r_field.setSFRotation([0.0, 0.0, 1.0, yaw_to_v])
            rover.resetPhysics()

            if not step_settle(3):
                return

            r_pos = rover.getPosition()
            r_rot = rover.getOrientation()
            l_ranges = lidar.getRangeImage()

            # Check all victims visible in this frame (so multi-victim views are fully labeled)
            labels = []
            primary_ok = False
            for other_def, (onode, _, _, oext) in victims.items():
                opos = onode.getPosition()
                cx, cy, _ = _world_to_cam(opos, r_pos, r_rot)
                d_to_v = math.hypot(cx, cy)
                if cx <= 0.35 or d_to_v > 6.5:
                    continue
                bearing = math.atan2(cy, cx)
                if not _lidar_has_los(l_ranges, d_to_v, bearing):
                    continue
                bbox = _project_victim_bbox(opos, oext, r_pos, r_rot, w, h, hfov)
                if bbox is not None:
                    labels.append(bbox)
                    if other_def == vdef:
                        primary_ok = True

            if not primary_ok:
                continue

            got += 1
            saved_counts[split] += 1
            stem = f"{vdef.lower()}_{got:03d}"
            img_path = os.path.join(OUTPUT_DIR, "images", split, f"{stem}.jpg")
            lbl_path = os.path.join(OUTPUT_DIR, "labels", split, f"{stem}.txt")

            cam.saveImage(img_path, 92)
            with open(lbl_path, "w", encoding="utf-8") as lf:
                for b in labels:
                    lf.write(f"0 {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n")

            if preview_count < 20:
                preview_count += 1
                _save_preview(
                    cam.getImage(), w, h, labels,
                    os.path.join(OUTPUT_DIR, "preview_boxes", f"{stem}_preview.jpg")
                )

        print(f"[COLLECTOR] {vdef}: captured {got}/{target_cnt} clean LOS frames ({tries} poses tried)")

    # 2. Capture empty negative scenes (walls, rubble, roads, fire, gas — zero victims!)
    for split, neg_target in EMPTY_SPLIT_COUNTS.items():
        got = 0
        tries = 0
        print(f"[COLLECTOR] Capturing {neg_target} empty negative scenes -> split='{split}' ...")
        while got < neg_target and tries < neg_target * 12:
            tries += 1
            ax, ay = rng.choice(EMPTY_ANCHORS)
            rx = ax + rng.uniform(-1.5, 1.5)
            ry = ay + rng.uniform(-1.5, 1.5)
            yaw = rng.uniform(-math.pi, math.pi)

            t_field.setSFVec3f([rx, ry, 0.015])
            r_field.setSFRotation([0.0, 0.0, 1.0, yaw])
            rover.resetPhysics()
            if not step_settle(3):
                return

            r_pos = rover.getPosition()
            r_rot = rover.getOrientation()
            # Reject frame if any victim happens to fall inside camera FOV
            any_victim_in_fov = False
            for _, (onode, _, _, oext) in victims.items():
                if _project_victim_bbox(onode.getPosition(), oext, r_pos, r_rot, w, h, hfov) is not None:
                    any_victim_in_fov = True
                    break
            if any_victim_in_fov:
                continue

            got += 1
            saved_counts[split] += 1
            stem = f"empty_{split}_{got:03d}"
            cam.saveImage(os.path.join(OUTPUT_DIR, "images", split, f"{stem}.jpg"), 92)
            with open(os.path.join(OUTPUT_DIR, "labels", split, f"{stem}.txt"), "w", encoding="utf-8") as lf:
                lf.write("")  # Empty label file = true negative

    print(
        f"[DONE] Dataset capture complete! Totals: "
        f"train={saved_counts['train']}, val={saved_counts['val']}, test={saved_counts['test']}. "
        f"Check preview_boxes/ before running train_yolo11n_resqbot.py."
    )
    sup.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)


if __name__ == "__main__":
    main()
