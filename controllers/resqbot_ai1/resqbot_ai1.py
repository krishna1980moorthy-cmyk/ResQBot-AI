"""ResQBot AI — main Webots controller (central coordinator).

Loop per control step:
  sense (LiDAR, RGB, thermal, gas) -> fuse -> navigation proposal
  -> mission state machine picks final command -> motors.update() -> metrics

Folder: controllers/resqbot_ai/   (Webots requires file name == folder name)
"""
import time
from controller import Robot

import config as C
from motor_control import MotorController
from navigation import Navigator
from rgb_detection import RGBDetector
from thermal_sensor import ThermalSensor
from gas_sensor import GasSensor
from sensor_fusion import SensorFusion
from mission_logic import Mission, STOP
from metrics import Metrics
from resqbot_live_monitor import ResQBotLiveMonitor


class ResQBotAI:
    def __init__(self):
        self.robot = Robot()
        self.ts = int(self.robot.getBasicTimeStep())
        print(f"[resqbot] init, timestep={self.ts} ms")

        self.motors = MotorController(self.robot)
        self.nav = Navigator(self.robot, self.ts)
        self.rgb = RGBDetector(self.robot, self.ts)
        self.thermal = ThermalSensor(self.robot, self.ts)
        self.gas = GasSensor()
        self.fusion = SensorFusion()
        self.mission = Mission(self.ts)
        self.metrics = Metrics()
        self.monitor = ResQBotLiveMonitor(self.robot, self.ts)

    def run(self):
        # let sensors produce first frames
        for _ in range(3):
            if self.robot.step(self.ts) == -1:
                return
        try:
            while self.robot.step(self.ts) != -1:
                t0 = time.perf_counter()

                pose = self.nav.pose()
                dets = self.rgb.detect()
                best = dets[0] if dets else None
                th = self.thermal.read()
                gas = self.gas.read(pose[0], pose[1])
                fused = self.fusion.fuse(best, th, gas, rgb_available=self.rgb.available)
                if fused.suspect:
                    fused.victim_xy = self.nav.project(fused.victim_bearing)

                nav_cmd = self.nav.decide(self.motors, commanding_motion=self.mission.state == "SEARCH")
                state = self.mission.step(fused, nav_cmd, pose, self.motors, self.fusion)

                # safety layer: never drive forward into an obstacle, whatever the state
                if self.nav.sectors["front"] < C.OBSTACLE_STOP_DIST * 0.6:
                    l, r = self.motors.target
                    if l + r > 0:
                        self.motors.stop(emergency=True)
                        state += "|ESTOP"

                self.motors.update()
                step_ms = (time.perf_counter() - t0) * 1000
                yolo_ms = self.rgb.last_infer_ms if self.rgb.step_i % C.YOLO_EVERY_N_STEPS == 0 else None
                self.metrics.log(self.mission.t, state, pose, self.nav.sectors, self.motors,
                                 best, th, fused, yolo_ms, step_ms)

                self.monitor.update(self.mission.t, self.rgb, dets, self.nav, pose, gas, th, fused, self.motors, self.mission)

                if self.mission.state == STOP:
                    break
        finally:
            self.monitor.close()
            self.motors.stop(emergency=True)
            self.motors.update()
            s = self.metrics.close(self.mission, self.nav)
            print(f"[resqbot] done: {s['victims_confirmed']} victims, "
                  f"{s['distance_m']} m, {s['sim_time_s']} s")


if __name__ == "__main__":
    ResQBotAI().run()
