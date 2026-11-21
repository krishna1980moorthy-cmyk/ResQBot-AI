"""RGB camera + YOLO11n person detection (Ultralytics).

Degrades gracefully: if ultralytics/numpy aren't installed or the camera is
missing, detect() returns no detections and `available` is False.
"""
import time
import config as C

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover
    YOLO = None


class Detection:
    __slots__ = ("conf", "cx", "cy", "w", "h", "bearing")

    def __init__(self, conf, cx, cy, w, h, bearing):
        self.conf, self.cx, self.cy, self.w, self.h, self.bearing = conf, cx, cy, w, h, bearing

    def __repr__(self):
        return f"Person(conf={self.conf:.2f}, bearing={self.bearing:+.2f}, area={self.w*self.h:.3f})"


class RGBDetector:
    def __init__(self, robot, timestep):
        self.cam = robot.getDevice(C.DEVICE_NAMES["rgb_camera"])
        self.model = None
        self.step_i = 0
        self.last = []
        self.last_infer_ms = 0.0
        if self.cam:
            self.cam.enable(timestep)
            self.W, self.H = self.cam.getWidth(), self.cam.getHeight()
            self.fov = self.cam.getFov()
        if self.cam and YOLO and np is not None:
            try:
                self.model = YOLO(C.YOLO_WEIGHTS)
                print(f"[rgb] YOLO loaded: {C.YOLO_WEIGHTS}")
            except Exception as e:
                print(f"[rgb] YOLO load failed: {e}")
        else:
            print("[rgb] YOLO disabled (missing camera / ultralytics / numpy)")

    @property
    def available(self):
        return self.model is not None

    def _frame(self):
        raw = self.cam.getImage()
        if not raw:
            return None
        img = np.frombuffer(raw, np.uint8).reshape((self.H, self.W, 4))
        return np.ascontiguousarray(img[:, :, :3])   # BGRA -> BGR

    def detect(self):
        """Returns list[Detection] (cached between inference steps)."""
        self.step_i += 1
        if not self.available or self.step_i % C.YOLO_EVERY_N_STEPS:
            return self.last
        frame = self._frame()
        if frame is None:
            return self.last
        t0 = time.perf_counter()
        res = self.model.predict(frame, imgsz=C.YOLO_IMGSZ, conf=C.YOLO_CONF,
                                 classes=[C.PERSON_CLASS_ID], verbose=False)[0]
        self.last_infer_ms = (time.perf_counter() - t0) * 1000
        dets = []
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            cx, cy = (x1 + x2) / 2 / self.W, (y1 + y2) / 2 / self.H
            bearing = (0.5 - cx) * self.fov          # +ve = person to the left
            dets.append(Detection(float(b.conf[0]), cx, cy,
                                  (x2 - x1) / self.W, (y2 - y1) / self.H, bearing))
        dets.sort(key=lambda d: d.conf, reverse=True)
        self.last = dets
        return dets

    def best(self):
        return self.last[0] if self.last else None
