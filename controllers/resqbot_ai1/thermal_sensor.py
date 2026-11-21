"""Simulated 32x24 thermal array (MLX90640-style).

Source priority:
  1. A Webots Camera named DEVICE_NAMES['thermal_camera'] — its grayscale
     brightness is mapped to °C. Give victim models a bright/emissive
     appearance (or use Webots' recognition colours) so they read "hot".
  2. Fallback: ambient field + Gaussian noise (no hot spots) — fusion then
     relies on RGB only, which is the honest behaviour.
"""
import random
import config as C

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


class ThermalFrame:
    def __init__(self, grid, hot_pixels, max_c, hot_cx, fov=1.0):
        self.grid = grid            # 24x32 list/array of °C
        self.fov = fov
        self.hot_pixels = hot_pixels
        self.max_c = max_c
        self.hot_cx = hot_cx        # normalised column centroid of hot pixels (or None)

    @property
    def bearing(self):
        """rad, +ve = hot region left of centre (None if no hot region)."""
        return None if self.hot_cx is None or self.score == 0 else (0.5 - self.hot_cx) * self.fov

    @property
    def score(self):
        """0..1 likelihood a warm body is in view."""
        if self.hot_pixels < C.THERMAL_MIN_HOT_PIXELS:
            return 0.0
        size = min(self.hot_pixels / 60.0, 1.0)
        temp = min(max((self.max_c - C.BODY_TEMP_MIN_C) / 6.0, 0.0), 1.0)
        return round(0.5 * size + 0.5 * temp, 3)


class ThermalSensor:
    def __init__(self, robot, timestep):
        self.rows, self.cols = C.THERMAL_SHAPE
        self.cam = robot.getDevice(C.DEVICE_NAMES["thermal_camera"])
        if self.cam:
            self.cam.enable(timestep)
            self.W, self.H = self.cam.getWidth(), self.cam.getHeight()
            self.fov = self.cam.getFov() if hasattr(self.cam, "getFov") else 1.0
        else:
            self.fov = 1.0
        self.mode = "camera" if (self.cam and np is not None) else "synthetic"
        print(f"[thermal] mode = {self.mode}")
        self.last = None

    def _from_camera(self):
        raw = self.cam.getImage()
        if not raw:
            return None
        img = np.frombuffer(raw, np.uint8).reshape((self.H, self.W, 4))[:, :, :3]
        gray = img.mean(axis=2)
        # block-average down to 24x32
        ys = np.linspace(0, self.H, self.rows + 1, dtype=int)
        xs = np.linspace(0, self.W, self.cols + 1, dtype=int)
        small = np.array([[gray[ys[r]:ys[r + 1], xs[c]:xs[c + 1]].mean()
                           for c in range(self.cols)] for r in range(self.rows)])
        temp = C.AMBIENT_C + (small / 255.0) * (C.BODY_TEMP_MAX_C + 2 - C.AMBIENT_C)
        temp += np.random.normal(0, C.THERMAL_NOISE_C, temp.shape)
        return temp

    def _synthetic(self):
        return [[C.AMBIENT_C + random.gauss(0, C.THERMAL_NOISE_C) for _ in range(self.cols)]
                for _ in range(self.rows)]

    def read(self):
        grid = self._from_camera() if self.mode == "camera" else None
        if grid is None:
            grid = self._synthetic()
        hot, mx, col_sum = 0, -1e9, 0.0
        for r in range(self.rows):
            for c in range(self.cols):
                v = float(grid[r][c])
                mx = max(mx, v)
                if C.BODY_TEMP_MIN_C <= v <= C.BODY_TEMP_MAX_C + 2:
                    hot += 1
                    col_sum += c
        cx = (col_sum / hot) / (self.cols - 1) if hot else None
        self.last = ThermalFrame(grid, hot, mx, cx, self.fov)
        return self.last
