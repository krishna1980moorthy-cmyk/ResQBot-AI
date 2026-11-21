"""Simulated gas sensor (MQ-series style ppm) from configured hazard zones."""
import math
import random
import config as C


class GasReading:
    def __init__(self, ppm):
        self.ppm = ppm

    @property
    def level(self):
        if self.ppm >= C.GAS_DANGER_PPM:
            return "DANGER"
        if self.ppm >= C.GAS_WARN_PPM:
            return "WARN"
        return "SAFE"

    @property
    def score(self):
        return round(min(self.ppm / C.GAS_DANGER_PPM, 1.0), 3)


class GasSensor:
    def __init__(self, alpha=0.2):
        self.alpha = alpha          # EMA smoothing — mimics sensor response lag
        self.ppm = C.GAS_BASELINE_PPM
        self.last = GasReading(self.ppm)

    def read(self, x, y):
        raw = C.GAS_BASELINE_PPM
        if x is not None and y is not None:
            for zx, zy, rad, peak in C.GAS_ZONES:
                d = math.hypot(x - zx, y - zy)
                raw += peak * math.exp(-(d / rad) ** 2)   # Gaussian plume
        raw *= 1 + random.gauss(0, 0.03)
        self.ppm = (1 - self.alpha) * self.ppm + self.alpha * max(raw, 0.0)
        self.last = GasReading(round(self.ppm, 1))
        return self.last
