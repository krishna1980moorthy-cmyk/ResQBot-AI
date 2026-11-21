"""Mission state machine: SEARCH → INSPECT → ALERT, with AVOID/HAZARD/STOP.

States
  SEARCH   explore with LiDAR navigation, watch fusion
  INSPECT  slow down, centre the suspect target, wait for confirmation
  ALERT    hold position, log victim, then resume SEARCH
  HAZARD   gas DANGER — back out and turn away
  STOP     mission complete / time limit
"""
import math
import config as C

SEARCH, INSPECT, ALERT, HAZARD, STOP = "SEARCH", "INSPECT", "ALERT", "HAZARD", "STOP"


class Mission:
    def __init__(self, timestep):
        self.dt = timestep / 1000.0
        self.state = SEARCH
        self.t = 0.0
        self.state_t = 0.0
        self.victims = []        # list of dicts
        self.hazards = []
        self.events = []
        self._cooldown_until = 0.0

    def _go(self, s, why=""):
        if s != self.state:
            self.events.append((round(self.t, 2), self.state, s, why))
            if C.VERBOSE:
                print(f"[mission] t={self.t:6.1f}s  {self.state} -> {s}  {why}")
            self.state, self.state_t = s, 0.0

    def _known_victim(self, xy):
        if xy is None:
            return False
        return any(math.hypot(xy[0] - v["vx"], xy[1] - v["vy"]) < C.DUPLICATE_VICTIM_RADIUS
                   for v in self.victims if v["vx"] is not None)

    def _known_hazard(self, x, y):
        if x is None:
            return False
        return any(math.hypot(x - h["x"], y - h["y"]) < 1.5 for h in self.hazards)

    def step(self, fusion, nav_cmd, pose, motors, fusion_engine):
        """Decides final motor command. nav_cmd = (lin, ang, nav_status)."""
        self.t += self.dt
        self.state_t += self.dt
        x, y, _ = pose
        lin, ang, nav_status = nav_cmd

        if self.t >= C.MISSION_TIME_LIMIT_S and self.state != STOP:
            self._go(STOP, "time limit")

        if self.state == STOP:
            motors.stop()
            return "STOP"

        # Gas danger overrides everything except ALERT hold
        if fusion.hazard == "DANGER" and self.state not in (ALERT, HAZARD):
            if not self._known_hazard(x, y):
                self.hazards.append({"t": round(self.t, 2), "x": x, "y": y, "ppm": fusion.gas_ppm})
            self._go(HAZARD, f"gas {fusion.gas_ppm}ppm")

        if self.state == HAZARD:
            if self.state_t < 1.5:
                motors.drive(-0.4, 0.0)
            elif self.state_t < 3.0:
                motors.drive(0.0, 0.6)
            else:
                self._go(SEARCH, "hazard escaped")
            return HAZARD

        if self.state == SEARCH:
            if fusion.suspect and self.t >= self._cooldown_until and not self._known_victim(fusion.victim_xy):
                fusion_engine.reset()
                self._go(INSPECT, f"suspect conf={fusion.victim_conf}")
            else:
                motors.drive(lin, ang)
                return f"SEARCH/{nav_status}"

        if self.state == INSPECT:
            if fusion.confirmed and self.state_t >= C.INSPECT_MIN_S:
                if self._known_victim(fusion.victim_xy):
                    self._cooldown_until = self.t + C.REINSPECT_COOLDOWN_S
                    self._go(SEARCH, "already-logged victim")
                    return INSPECT
                vx, vy = fusion.victim_xy if fusion.victim_xy else (None, None)
                self.victims.append({"t": round(self.t, 2),
                                     "robot_x": None if x is None else round(x, 2),
                                     "robot_y": None if y is None else round(y, 2),
                                     "vx": None if vx is None else round(vx, 2),
                                     "vy": None if vy is None else round(vy, 2),
                                     "conf": fusion.victim_conf, "sources": fusion.sources,
                                     "gas": fusion.gas_ppm})
                self._go(ALERT, f"VICTIM CONFIRMED #{len(self.victims)}")
            elif self.state_t > C.INSPECT_TIMEOUT_S or (not fusion.suspect and self.state_t > 1.5):
                self._cooldown_until = self.t + C.REINSPECT_COOLDOWN_S
                self._go(SEARCH, "inspect timeout / lost target")
            else:
                # centre on target; creep forward only if path is clear
                turn = 0.0 if fusion.victim_bearing is None else max(-0.4, min(0.4, 1.2 * fusion.victim_bearing))
                fwd = C.INSPECT_SPEED if nav_status in ("CRUISE",) else 0.0
                motors.drive(fwd, turn)
            return INSPECT

        if self.state == ALERT:
            motors.stop()
            if self.state_t >= C.ALERT_HOLD_S:
                self._cooldown_until = self.t + C.REINSPECT_COOLDOWN_S
                self._go(SEARCH, "alert sent, resuming")
            return ALERT

        return self.state
