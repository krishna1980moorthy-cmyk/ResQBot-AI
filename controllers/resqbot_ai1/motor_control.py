"""4WD skid-steer motor control with soft acceleration ramp."""
import config as C


class MotorController:
    def __init__(self, robot):
        self.motors = {}
        for key, name in C.DEVICE_NAMES["motors"].items():
            m = robot.getDevice(name)
            if m is None:
                raise RuntimeError(f"[motor] device '{name}' not found — check config.DEVICE_NAMES")
            m.setPosition(float("inf"))   # velocity-control mode
            m.setVelocity(0.0)
            self.motors[key] = m
        self._left = 0.0   # current fraction [-1, 1]
        self._right = 0.0
        self.target = (0.0, 0.0)

    @staticmethod
    def _clamp(v, lo=-1.0, hi=1.0):
        return max(lo, min(hi, v))

    def _ramp(self, cur, tgt):
        d = tgt - cur
        if abs(d) > C.ACCEL_LIMIT:
            d = C.ACCEL_LIMIT if d > 0 else -C.ACCEL_LIMIT
        return cur + d

    def set_tracks(self, left, right, immediate=False):
        """left/right as fraction of MAX_WHEEL_SPEED in [-1, 1]."""
        self.target = (self._clamp(left), self._clamp(right))
        if immediate:
            self._left, self._right = self.target

    def drive(self, linear, angular):
        """linear, angular in [-1, 1]; positive angular = turn left."""
        self.set_tracks(linear - angular, linear + angular)

    # convenience primitives
    def forward(self, s=C.CRUISE_SPEED): self.set_tracks(s, s)
    def reverse(self, s=C.CRUISE_SPEED): self.set_tracks(-s, -s)
    def turn_left(self, s=C.TURN_SPEED): self.set_tracks(-s, s)
    def turn_right(self, s=C.TURN_SPEED): self.set_tracks(s, -s)

    def stop(self, emergency=False):
        self.set_tracks(0.0, 0.0, immediate=emergency)

    def update(self):
        """Call once per control step — applies ramped velocities."""
        self._left = self._ramp(self._left, self.target[0])
        self._right = self._ramp(self._right, self.target[1])
        for key, m in self.motors.items():
            side = self._left if "left" in key else self._right
            m.setVelocity(side * C.MAX_WHEEL_SPEED * C.MOTOR_DIRECTION[key])

    @property
    def is_moving(self):
        return abs(self._left) + abs(self._right) > 0.05

    @property
    def state(self):
        return round(self._left, 3), round(self._right, 3)
