import math

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None


class ResQBotLiveMonitor:
    def __init__(self, robot, timestep):
        self.next_log = 0.0
        self.next_view = 0.0
        self.victim_count = 0
        self.window = "ResQBot AI | RGB + Thermal Proxy"
        self.window_ok = cv2 is not None and np is not None
        self.extra = {}

        names = [
            "imu_accel", "imu_gyro",
            "front_center_ds", "front_left_ds",
            "front_right_ds", "rear_left_ds",
            "rear_right_ds",
            "front_left_encoder", "front_right_encoder",
            "rear_left_encoder", "rear_right_encoder"
        ]

        for name in names:
            dev = robot.getDevice(name)
            if dev is not None:
                dev.enable(timestep)
            self.extra[name] = dev

        print("[MONITOR] Live telemetry enabled", flush=True)

    @staticmethod
    def fmt(value):
        try:
            v = float(value)
            return f"{v:.2f}" if math.isfinite(v) else "N/A"
        except (TypeError, ValueError):
            return "N/A"

    def val(self, name):
        dev = self.extra.get(name)
        return self.fmt(dev.getValue()) if dev else "N/A"

    def vec(self, name):
        dev = self.extra.get(name)
        if not dev:
            return "N/A"
        return "(" + ",".join(
            self.fmt(x) for x in dev.getValues()
        ) + ")"

    def update(self, t, rgb, dets, nav, pose,
               gas, thermal, fused, motors, mission):

        # Print only newly confirmed victims
        for v in mission.victims[self.victim_count:]:
            valid_pose = all(x is not None for x in pose)
            valid_xy = (
                v.get("vx") is not None and
                v.get("vy") is not None
            )

            if valid_pose and valid_xy:
                loc = (
                    f"APPROX X={v['vx']:.2f}m "
                    f"Y={v['vy']:.2f}m"
                )
            else:
                loc = "LOCATION UNKNOWN (GPS/yaw missing)"

            print(
                f"\n[VICTIM CONFIRMED] t={t:.1f}s "
                f"{loc} | conf={v['conf']:.2f} "
                f"| sources={v['sources']}\n",
                flush=True
            )

        self.victim_count = len(mission.victims)
        best = dets[0] if dets else None

        # Console sensor values once per simulation second
        if t >= self.next_log:
            self.next_log = t + 1.0

            gps = (
                nav.gps.getValues() if nav.gps
                else None
            )
            rpy = (
                nav.imu.getRollPitchYaw() if nav.imu
                else None
            )

            gps_text = (
                "(" + ",".join(self.fmt(v) for v in gps) + ")"
                if gps else "N/A"
            )
            rpy_text = (
                "(" + ",".join(self.fmt(v) for v in rpy) + ")"
                if rpy else "N/A"
            )

            print(
                f"\n[SENSORS] t={t:.1f}s "
                f"STATE={mission.state}"
            )
            print(
                f"[GPS] XYZ={gps_text} "
                f"[IMU] Roll/Pitch/Yaw={rpy_text}"
            )
            print(
                f"[LIDAR] Front={self.fmt(nav.sectors['front'])}m "
                f"Left={self.fmt(nav.sectors['left'])}m "
                f"Right={self.fmt(nav.sectors['right'])}m"
            )
            print(
                f"[GAS SIM] {self.fmt(gas.ppm)} ppm "
                f"Status={gas.level}"
            )
            print(
                f"[THERMAL PROXY] Max={self.fmt(thermal.max_c)}C "
                f"HotPixels={thermal.hot_pixels} "
                f"Score={self.fmt(thermal.score)}"
            )
            print(
                f"[RGB YOLO] Persons={len(dets)} "
                f"BestConf={self.fmt(best.conf if best else None)} "
                f"Fusion={self.fmt(fused.victim_conf)}"
            )
            print(
                f"[IMU RAW] Accel={self.vec('imu_accel')} "
                f"Gyro={self.vec('imu_gyro')}"
            )
            print(
                "[DISTANCE SENSOR RAW] " +
                " ".join(
                    f"{short}={self.val(name)}"
                    for short, name in [
                        ("FC", "front_center_ds"),
                        ("FL", "front_left_ds"),
                        ("FR", "front_right_ds"),
                        ("RL", "rear_left_ds"),
                        ("RR", "rear_right_ds")
                    ]
                )
            )
            print(
                "[WHEEL ENCODERS rad] " +
                " ".join(
                    f"{short}={self.val(name)}"
                    for short, name in [
                        ("FL", "front_left_encoder"),
                        ("FR", "front_right_encoder"),
                        ("RL", "rear_left_encoder"),
                        ("RR", "rear_right_encoder")
                    ]
                )
            )
            print(f"[MOTORS] Command L/R={motors.state}")

            if best:
                xy = (
                    nav.project(best.bearing)
                    if all(x is not None for x in pose)
                    else None
                )
                loc = (
                    f"APPROX X={xy[0]:.2f} Y={xy[1]:.2f}m"
                    if xy and all(x is not None for x in xy)
                    else "location UNKNOWN"
                )
                print(
                    f"[PERSON CANDIDATE] "
                    f"YOLO={best.conf:.2f} "
                    f"Bearing={best.bearing:+.2f}rad "
                    f"{loc}",
                    flush=True
                )

        # Camera + annotated YOLO view (4 FPS simulation time)
        if not self.window_ok or t < self.next_view:
            return
        self.next_view = t + 0.25

        try:
            raw = rgb.cam.getImage() if rgb.cam else None
            if not raw:
                return

            w = rgb.cam.getWidth()
            h = rgb.cam.getHeight()

            img = np.frombuffer(
                raw, dtype=np.uint8
            ).reshape(h, w, 4)[:, :, :3].copy()

            for d in dets:
                x1 = max(0, int((d.cx-d.w/2)*w))
                y1 = max(0, int((d.cy-d.h/2)*h))
                x2 = min(w-1, int((d.cx+d.w/2)*w))
                y2 = min(h-1, int((d.cy+d.h/2)*h))

                cv2.rectangle(
                    img, (x1,y1), (x2,y2),
                    (0,255,0), 2
                )
                cv2.putText(
                    img, f"PERSON {d.conf:.2f}",
                    (x1,max(15,y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,(0,255,0),1
                )

            rgb_panel = cv2.resize(img, (400,300))

            grid = np.array(
                thermal.grid, dtype=np.float32
            )
            pseudo = np.uint8(
                np.clip((grid-25)*13,0,255)
            )
            heat_panel = cv2.resize(
                cv2.applyColorMap(
                    pseudo, cv2.COLORMAP_JET
                ),
                (400,300),
                interpolation=cv2.INTER_NEAREST
            )

            screen = np.zeros(
                (385,800,3), dtype=np.uint8
            )
            screen[:300,:400] = rgb_panel
            screen[:300,400:] = heat_panel

            cv2.putText(
                screen, "RGB + YOLO11n", (8,20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,(255,255,255),2
            )
            cv2.putText(
                screen, "THERMAL PROXY - NOT REAL HEAT",
                (408,20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,(255,255,255),1
            )

            lines = [
                f"State={mission.state} | GPS X={self.fmt(pose[0])} "
                f"Y={self.fmt(pose[1])} | Yaw={self.fmt(pose[2])}",
                f"LiDAR F/L/R: "
                f"{self.fmt(nav.sectors['front'])} / "
                f"{self.fmt(nav.sectors['left'])} / "
                f"{self.fmt(nav.sectors['right'])} m",
                f"Persons={len(dets)} | "
                f"GasSim={self.fmt(gas.ppm)}ppm | "
                f"Victims={len(mission.victims)}"
            ]

            for i, line in enumerate(lines):
                cv2.putText(
                    screen, line, (8,323+24*i),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.53,(200,255,200),1
                )

            cv2.imshow(self.window, screen)
            cv2.waitKey(1)

        except Exception as e:
            self.window_ok = False
            print(
                f"[MONITOR] Camera window unavailable: {e}",
                flush=True
            )

    def close(self):
        if self.window_ok:
            try:
                cv2.destroyWindow(self.window)
            except Exception:
                pass