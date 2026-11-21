"""LiDAR obstacle avoidance, route decisions and stuck recovery."""
import math
import config as C


class Navigator:
    def __init__(self, robot, timestep):
        self.dt = timestep / 1000.0
        self.lidar = robot.getDevice(C.DEVICE_NAMES["lidar"])
        if self.lidar is None:
            raise RuntimeError("[nav] lidar not found — check config.DEVICE_NAMES['lidar']")
        self.lidar.enable(timestep)
        self.gps = robot.getDevice(C.DEVICE_NAMES["gps"])
        if self.gps:
            self.gps.enable(timestep)
        self.imu = robot.getDevice(C.DEVICE_NAMES["imu"])
        if self.imu:
            self.imu.enable(timestep)

        self.sectors = {"front": math.inf, "left": math.inf, "right": math.inf}
        self._hist = []            # (t, x, y) samples for stuck detection
        self._recover_t = 0.0      # remaining recovery time
        self._recover_dir = 1
        self.t = 0.0
        self.distance_travelled = 0.0
        self._last_pos = None

    # ---------------------------------------------------------- sensing --
    def pose(self):
        x = y = yaw = None
        if self.gps:
            v = self.gps.getValues()
            if all(math.isfinite(a) for a in v):
                x, y = v[0], v[1]
        if self.imu:
            yaw = self.imu.getRollPitchYaw()[2]
        return x, y, yaw

    def _sector_min(self, ranges, fov, c_deg, w_deg):
        n = len(ranges)
        if n == 0:
            return math.inf
        half = math.radians(w_deg) / 2
        c = math.radians(c_deg)
        best = math.inf
        for i, r in enumerate(ranges):
            # Webots lidar: index 0 = leftmost (+fov/2), last = rightmost (-fov/2)
            ang = fov / 2 - i * fov / max(n - 1, 1)
            if abs(ang - c) <= half and math.isfinite(r) and r > 0.02:
                best = min(best, r)
        return best

    def sense(self):
        ranges = self.lidar.getRangeImage() or []
        fov = self.lidar.getFov()
        side_c = min(math.degrees(fov) / 2 - C.SIDE_SECTOR_DEG / 2, 70)
        self.sectors["front"] = self._sector_min(ranges, fov, 0, C.FRONT_SECTOR_DEG)
        self.sectors["left"] = self._sector_min(ranges, fov, side_c, C.SIDE_SECTOR_DEG)
        self.sectors["right"] = self._sector_min(ranges, fov, -side_c, C.SIDE_SECTOR_DEG)
        return self.sectors

    # ---------------------------------------------------------- stuck ----
    def _update_stuck(self, commanding_motion):
        self.t += self.dt
        x, y, _ = self.pose()
        if x is None:
            return False
        if self._last_pos:
            self.distance_travelled += math.hypot(x - self._last_pos[0], y - self._last_pos[1])
        self._last_pos = (x, y)
        if not commanding_motion:
            self._hist.clear()
            return False
        self._hist.append((self.t, x, y))
        while self._hist and self.t - self._hist[0][0] > C.STUCK_WINDOW_S:
            self._hist.pop(0)
        if self.t - self._hist[0][0] < C.STUCK_WINDOW_S * 0.9:
            return False
        t0, x0, y0 = self._hist[0]
        return math.hypot(x - x0, y - y0) < C.STUCK_MIN_DISP

    # --------------------------------------------------------- decide ----
    def decide(self, motors, commanding_motion=True):
        """Returns (linear, angular, status) — caller applies via motors.drive()."""
        s = self.sense()
        if self._recover_t > 0:
            self._recover_t -= self.dt
            if self._recover_t > C.RECOVERY_TURN_S:
                return -0.4, 0.0, "RECOVER_REVERSE"
            return 0.0, 0.6 * self._recover_dir, "RECOVER_TURN"

        if self._update_stuck(commanding_motion and motors.is_moving):
            self._recover_t = C.RECOVERY_REVERSE_S + C.RECOVERY_TURN_S
            self._recover_dir = 1 if s["left"] > s["right"] else -1
            self._hist.clear()
            return -0.4, 0.0, "STUCK"

        f = s["front"]
        if f < C.OBSTACLE_STOP_DIST:
            turn = 1 if s["left"] > s["right"] else -1
            return 0.0, C.TURN_SPEED * turn, "BLOCKED_TURN"
        if f < C.OBSTACLE_AVOID_DIST:
            k = (C.OBSTACLE_AVOID_DIST - f) / (C.OBSTACLE_AVOID_DIST - C.OBSTACLE_STOP_DIST)
            turn = 1 if s["left"] > s["right"] else -1
            return C.CRUISE_SPEED * (1 - 0.7 * k), 0.5 * k * turn, "AVOID"

        # wall-balancing: drift away from the closer side
        l, r = min(s["left"], 2.0), min(s["right"], 2.0)
        bias = 0.15 * (l - r) / 2.0
        return C.CRUISE_SPEED, bias, "CRUISE"

    def range_at(self, bearing, width_deg=6):
        ranges = self.lidar.getRangeImage() or []
        return self._sector_min(ranges, self.lidar.getFov(), math.degrees(bearing), width_deg)

    def project(self, bearing):
        """World (x, y) of a target seen at `bearing` (rad, +left). None if no pose."""
        x, y, yaw = self.pose()
        if x is None:
            return None
        if bearing is None or yaw is None:
            return (x, y)
        d = self.range_at(bearing)
        if not math.isfinite(d) or d > 4.0:
            d = C.VICTIM_DEFAULT_RANGE_M
        return (x + d * math.cos(yaw + bearing), y + d * math.sin(yaw + bearing))

    def front_clear(self):
        return self.sectors["front"] > C.OBSTACLE_AVOID_DIST
