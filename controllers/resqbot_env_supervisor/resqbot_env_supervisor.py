"""
resqbot_env_supervisor.py — Isolated Simulation Environment Sensor Generator.

Purpose & Strict Separation:
  - Runs on the separate invisible `ResQBot_Env_Supervisor` node (supervisor TRUE).
  - Reads scene ground-truth node positions (`ResQBot_AI`, `Victim_01`..`05`,
    `Fire_Heat_Source_01`..`02`, `Gas_Leak_Source`) ONLY to synthesize realistic
    local sensor observations:
      1. Simulated MLX90640 32x24 thermal matrix (pinhole projection + range
         attenuation + Gaussian noise + line-of-sight distance cutoff)
      2. Simulated local MQ-2 gas concentration (ppm) at the rover's position
         (inverse-square spatial decay + downwind drift bias + sensor noise)
  - Transmits ONLY these noisy sensor observations on Emitter channel 1.
  - The autonomous rover (`resqbot_ai`) NEVER has Supervisor access and NEVER
    receives victim/hazard world coordinates or ground-truth labels.
"""

from __future__ import annotations

import json
import math
import random
from typing import List, Tuple

from controller import Supervisor


THERMAL_W = 32
THERMAL_H = 24
HFOV_RAD = 1.10
VFOV_RAD = 0.84
MAX_THERMAL_RANGE_M = 8.5
AMBIENT_TEMP_C = 23.5

VICTIM_DEFS = ("Victim_01", "Victim_02", "Victim_03", "Victim_04", "Victim_05")
FIRE_DEFS = ("Fire_Heat_Source_01", "Fire_Heat_Source_02")
GAS_SOURCE_DEF = "Gas_Leak_Source"


def _world_to_camera(
    target_xyz: List[float],
    rover_xyz: List[float],
    rover_rot_3x3: List[float],
) -> Tuple[float, float, float]:
    """Transform world point into rover-local frame (+X forward, +Y left, +Z up)."""
    dx = target_xyz[0] - rover_xyz[0]
    dy = target_xyz[1] - rover_xyz[1]
    dz = target_xyz[2] - rover_xyz[2]
    # R is row-major 3x3 (local -> world); transpose maps world -> local
    lx = rover_rot_3x3[0] * dx + rover_rot_3x3[3] * dy + rover_rot_3x3[6] * dz
    ly = rover_rot_3x3[1] * dx + rover_rot_3x3[4] * dy + rover_rot_3x3[7] * dz
    lz = rover_rot_3x3[2] * dx + rover_rot_3x3[5] * dy + rover_rot_3x3[8] * dz
    # Camera offset on ResQBotRover.proto: (0.215, 0.0, 0.194)
    return lx - 0.215, ly, lz - 0.194


def _render_thermal_frame(
    rover_xyz: List[float],
    rover_rot: List[float],
    victim_pts: List[List[float]],
    fire_pts: List[List[float]],
    rng: random.Random,
) -> List[float]:
    """Generate a noisy 32x24 simulated MLX90640 temperature matrix (deg C)."""
    frame = [AMBIENT_TEMP_C + rng.gauss(0.0, 0.55) for _ in range(THERMAL_W * THERMAL_H)]

    def splat_source(src_xyz: List[float], base_temp_c: float, radius_m: float) -> None:
        cx, cy, cz = _world_to_camera(src_xyz, rover_xyz, rover_rot)
        if cx <= 0.20 or cx > MAX_THERMAL_RANGE_M:
            return
        az = math.atan2(cy, cx)   # +left, -right
        el = math.atan2(cz, cx)   # +up, -down
        if abs(az) > HFOV_RAD * 0.55 or abs(el) > VFOV_RAD * 0.55:
            return
        # Map azimuth/elevation to pixel coordinates (x=0 left, y=0 top)
        u = (0.5 - az / HFOV_RAD) * THERMAL_W
        v = (0.5 - el / VFOV_RAD) * THERMAL_H
        ang_r_px = max(1.0, (radius_m / cx) * (THERMAL_W / HFOV_RAD))
        # Distance attenuation models MLX90640 pixel-averaging at longer range
        atten = max(0.35, 1.0 - (cx / (MAX_THERMAL_RANGE_M * 1.35)))
        peak = AMBIENT_TEMP_C + (base_temp_c - AMBIENT_TEMP_C) * atten

        x0 = max(0, int(u - ang_r_px * 1.5))
        x1 = min(THERMAL_W - 1, int(u + ang_r_px * 1.5))
        y0 = max(0, int(v - ang_r_px * 1.5))
        y1 = min(THERMAL_H - 1, int(v + ang_r_px * 1.5))
        for py in range(y0, y1 + 1):
            for px in range(x0, x1 + 1):
                d2 = (px + 0.5 - u) ** 2 + (py + 0.5 - v) ** 2
                w = math.exp(-d2 / max(0.6, 2.0 * (ang_r_px * 0.55) ** 2))
                val = AMBIENT_TEMP_C + (peak - AMBIENT_TEMP_C) * w
                idx = py * THERMAL_W + px
                if val > frame[idx]:
                    frame[idx] = val + rng.gauss(0.0, 0.35)

    for vp in victim_pts:
        splat_source([vp[0], vp[1], vp[2] + 0.22], base_temp_c=36.8, radius_m=0.35)
    for fp in fire_pts:
        splat_source([fp[0], fp[1], fp[2] + 0.45], base_temp_c=145.0, radius_m=0.75)

    return [round(t, 2) for t in frame]


def _compute_local_gas_ppm(
    rover_xyz: List[float],
    gas_src_xyz: List[float],
    rng: random.Random,
) -> float:
    """Distance-dependent local gas concentration (ppm) with mild SE plume drift + noise."""
    dx = rover_xyz[0] - gas_src_xyz[0]
    dy = rover_xyz[1] - gas_src_xyz[1]
    # Plume center drifts slightly SSE (~1.2m) matching Gas_Plume_Visual in world
    eff_d = math.hypot(dx - 0.8, dy + 0.8)
    baseline = 18.0
    plume = 950.0 * math.exp(-(eff_d * eff_d) / 14.0)
    return round(max(0.0, baseline + plume + rng.gauss(0.0, 4.5)), 1)


def main() -> None:
    sup = Supervisor()
    ts = int(sup.getBasicTimeStep())
    emitter = sup.getDevice("env_sensor_tx")
    rover_node = sup.getFromDef("ResQBot_AI")

    victim_nodes = [n for n in (sup.getFromDef(d) for d in VICTIM_DEFS) if n is not None]
    fire_nodes = [n for n in (sup.getFromDef(d) for d in FIRE_DEFS) if n is not None]
    gas_node = sup.getFromDef(GAS_SOURCE_DEF)

    rng = random.Random(42)
    step_i = 0
    print(
        f"[ENV_SUPERVISOR] Active | Rover={'OK' if rover_node else 'MISSING'} "
        f"Victims={len(victim_nodes)} Fires={len(fire_nodes)} Gas={'OK' if gas_node else 'NONE'}"
    )

    while sup.step(ts) != -1:
        step_i += 1
        # Publish simulated thermal & gas readings at ~5 Hz
        if step_i % 6 != 0 or emitter is None or rover_node is None:
            continue

        r_pos = rover_node.getPosition()
        r_rot = rover_node.getOrientation()
        v_pts = [n.getPosition() for n in victim_nodes]
        f_pts = [n.getPosition() for n in fire_nodes]
        g_pt = gas_node.getPosition() if gas_node is not None else [9.5, 13.45, 0.92]

        thermal_frame = _render_thermal_frame(r_pos, r_rot, v_pts, f_pts, rng)
        gas_ppm = _compute_local_gas_ppm(r_pos, g_pt, rng)

        payload = json.dumps({"thermal_32x24": thermal_frame, "gas_ppm": gas_ppm})
        emitter.send(payload.encode("utf-8"))


if __name__ == "__main__":
    main()
