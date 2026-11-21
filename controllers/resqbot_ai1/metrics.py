"""Logs real simulation performance to CSV + JSON summary (no fabricated numbers)."""
import csv
import json
import os
import time
import config as C


class Metrics:
    FIELDS = ["t", "state", "x", "y", "yaw", "front_m", "left_m", "right_m",
              "motor_l", "motor_r", "rgb_conf", "thermal_score", "hot_px",
              "victim_conf", "gas_ppm", "hazard", "yolo_ms", "step_ms"]

    def __init__(self):
        here = os.path.dirname(os.path.abspath(__file__))
        self.dir = os.path.join(here, C.LOG_DIR)
        os.makedirs(self.dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(self.dir, f"run_{stamp}.csv")
        self.json_path = os.path.join(self.dir, f"summary_{stamp}.json")
        self._f = open(self.csv_path, "w", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=self.FIELDS)
        self._w.writeheader()
        self.n = 0
        self.step_ms = []
        self.yolo_ms = []
        self.min_clearance = float("inf")
        self.collisions = 0      # front range < 0.10 m counted as contact
        self._in_contact = False

    def log(self, t, state, pose, sectors, motors, rgb_det, thermal, fusion, yolo_ms, step_ms):
        self.n += 1
        self.step_ms.append(step_ms)
        if yolo_ms:
            self.yolo_ms.append(yolo_ms)
        f = sectors["front"]
        if f != float("inf"):
            self.min_clearance = min(self.min_clearance, f)
        contact = f < 0.10
        if contact and not self._in_contact:
            self.collisions += 1
        self._in_contact = contact
        if self.n % C.LOG_EVERY_N_STEPS:
            return
        r = lambda v: None if v is None or v == float("inf") else round(v, 3)
        self._w.writerow({
            "t": round(t, 3), "state": state, "x": r(pose[0]), "y": r(pose[1]), "yaw": r(pose[2]),
            "front_m": r(sectors["front"]), "left_m": r(sectors["left"]), "right_m": r(sectors["right"]),
            "motor_l": motors.state[0], "motor_r": motors.state[1],
            "rgb_conf": r(rgb_det.conf) if rgb_det else 0.0,
            "thermal_score": thermal.score, "hot_px": thermal.hot_pixels,
            "victim_conf": fusion.victim_conf, "gas_ppm": fusion.gas_ppm, "hazard": fusion.hazard,
            "yolo_ms": r(yolo_ms), "step_ms": r(step_ms),
        })

    def close(self, mission, nav):
        self._f.close()
        avg = lambda a: round(sum(a) / len(a), 2) if a else None
        summary = {
            "sim_time_s": round(mission.t, 2),
            "steps": self.n,
            "distance_m": round(nav.distance_travelled, 2),
            "victims_confirmed": len(mission.victims),
            "victims": mission.victims,
            "gas_hazards": mission.hazards,
            "collisions_est": self.collisions,
            "min_front_clearance_m": None if self.min_clearance == float("inf") else round(self.min_clearance, 3),
            "avg_step_ms": avg(self.step_ms),
            "avg_yolo_ms": avg(self.yolo_ms),
            "state_transitions": mission.events,
        }
        with open(self.json_path, "w") as fp:
            json.dump(summary, fp, indent=2)
        print(f"[metrics] saved {self.csv_path}\n[metrics] saved {self.json_path}")
        return summary
