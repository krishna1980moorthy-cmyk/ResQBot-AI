"""Combines RGB + thermal into a victim confidence, and gas into a hazard state.

Late fusion with geometric agreement: thermal only boosts RGB when the hot
region lines up horizontally with the YOLO box (or no RGB box exists).
"""
import config as C


class FusionResult:
    def __init__(self, victim_conf, victim_bearing, hazard, gas_ppm, confirmed, sources):
        self.victim_conf = victim_conf
        self.victim_bearing = victim_bearing
        self.hazard = hazard            # SAFE / WARN / DANGER
        self.gas_ppm = gas_ppm
        self.confirmed = confirmed
        self.sources = sources          # e.g. "rgb+thermal"
        self.victim_xy = None           # filled by main controller (world estimate)

    @property
    def suspect(self):
        return self.victim_conf >= C.FUSION_VICTIM_SUSPECT

    def __repr__(self):
        return (f"Fusion(victim={self.victim_conf:.2f} [{self.sources}] "
                f"confirmed={self.confirmed} hazard={self.hazard} gas={self.gas_ppm}ppm)")


class SensorFusion:
    def __init__(self):
        self._streak = 0

    def fuse(self, rgb_det, thermal, gas, rgb_available=True):
        rgb = rgb_det.conf if rgb_det else 0.0
        th = thermal.score if thermal else 0.0

        agree = 1.0
        if rgb_det and thermal and thermal.hot_cx is not None:
            dx = abs(rgb_det.cx - thermal.hot_cx)
            agree = max(0.0, 1.0 - dx / 0.35)   # 0 if >35% of frame apart
        th_eff = th * agree if rgb_det else th

        if not rgb_available and C.THERMAL_ONLY_FALLBACK:
            conf = th_eff                       # renormalise: thermal is the only vision cue
        else:
            conf = C.W_RGB * rgb + C.W_THERMAL * th_eff
        if rgb > 0 and th_eff > 0:
            conf = min(1.0, conf + 0.15 * min(rgb, th_eff))   # agreement bonus

        sources = "+".join(n for n, v in (("rgb", rgb), ("thermal", th_eff)) if v > 0) or "none"
        # RESQBOT_RGB_REQUIRED_GATE
        # Thermal camera brightness alone is NOT victim evidence.
        if rgb_det is None:
            conf = 0.0
            sources = "none"
        self._streak = self._streak + 1 if conf >= C.FUSION_VICTIM_CONFIRM else 0
        confirmed = self._streak >= C.CONFIRM_FRAMES

        bearing = rgb_det.bearing if rgb_det else (thermal.bearing if thermal else None)
        return FusionResult(round(conf, 3), bearing, gas.level, gas.ppm, confirmed, sources)

    def reset(self):
        self._streak = 0
