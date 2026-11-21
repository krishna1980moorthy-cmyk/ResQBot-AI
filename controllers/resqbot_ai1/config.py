"""ResQBot AI â€” central configuration.

Every device name, threshold and tuning constant lives here so modules never
hard-code values. Edit DEVICE_NAMES to match the names in your custom rover PROTO.
"""

# ---------------------------------------------------------------- devices ----
DEVICE_NAMES = {
    # 4WD wheel motors (RotationalMotor names in the rover PROTO)
    "motors": {
        "front_left": "front_left_motor",
        "front_right": "front_right_motor",
        "rear_left": "rear_left_motor",
        "rear_right": "rear_right_motor",
    },
    "lidar": "lidar",
    "rgb_camera": "rgb_camera",
    "thermal_camera": "thermal_camera",   # optional: a low-res Camera used as thermal proxy
    "gps": "gps",                         # optional but recommended
    "imu": "inertial_unit",               # optional
}

# Flip a side if wheels spin backwards in your PROTO
MOTOR_DIRECTION = {"front_left": 1, "front_right": 1, "rear_left": 1, "rear_right": 1}

# --------------------------------------------------------------- motion -----
MAX_WHEEL_SPEED = 10.0        # rad/s (keep <= motor maxVelocity in PROTO)
CRUISE_SPEED = 0.6            # fraction of MAX_WHEEL_SPEED
TURN_SPEED = 0.5
INSPECT_SPEED = 0.25
ACCEL_LIMIT = 0.08            # max fraction change per control step (soft ramp)

# ------------------------------------------------------------ navigation ----
OBSTACLE_STOP_DIST = 0.35     # m â€” emergency stop inside this range (front sector)
OBSTACLE_AVOID_DIST = 0.80    # m â€” start steering away
FRONT_SECTOR_DEG = 40         # total width of front sector
SIDE_SECTOR_DEG = 60
STUCK_WINDOW_S = 4.0          # seconds of commanded motion with no displacement
STUCK_MIN_DISP = 0.05         # m
RECOVERY_REVERSE_S = 1.2
RECOVERY_TURN_S = 1.0

# -------------------------------------------------------------- vision ------
YOLO_WEIGHTS = "resqbot_victim_best.pt"
YOLO_CONF = 0.20
YOLO_EVERY_N_STEPS = 4        # run inference every N control steps
YOLO_IMGSZ = 640
PERSON_CLASS_ID = 0           # COCO "person"

# ------------------------------------------------------------- thermal ------
THERMAL_SHAPE = (24, 32)      # rows, cols (MLX90640-style)
AMBIENT_C = 26.0
BODY_TEMP_MIN_C = 30.0        # pixel counted as "warm body" above this
BODY_TEMP_MAX_C = 40.0
THERMAL_MIN_HOT_PIXELS = 6
THERMAL_NOISE_C = 0.4

# ----------------------------------------------------------------- gas ------
# Simulated hazard zones in world coordinates (x, y, radius_m, peak_ppm)
GAS_ZONES = [
    (2.0, -1.5, 1.2, 400.0),
    (-3.0, 2.5, 1.5, 900.0),
]
GAS_BASELINE_PPM = 5.0
GAS_WARN_PPM = 100.0
GAS_DANGER_PPM = 300.0

# --------------------------------------------------------------- fusion -----
FUSION_VICTIM_CONFIRM = 0.70  # fused confidence to raise victim alert
FUSION_VICTIM_SUSPECT = 0.40  # start INSPECT
W_RGB = 0.55
W_THERMAL = 0.45
CONFIRM_FRAMES = 3            # consecutive frames above confirm threshold

# -------------------------------------------------------------- mission -----
INSPECT_TIMEOUT_S = 6.0
INSPECT_MIN_S = 1.0           # minimum dwell before a victim can be confirmed
VICTIM_DEFAULT_RANGE_M = 1.5  # used when LiDAR has no return along the bearing
ALERT_HOLD_S = 3.0
MISSION_TIME_LIMIT_S = 600.0
DUPLICATE_VICTIM_RADIUS = 1.5 # m â€” ignore re-detections near a logged victim
REINSPECT_COOLDOWN_S = 5.0    # after a failed inspect, ignore suspects for this long
THERMAL_ONLY_FALLBACK = False  # if YOLO unavailable, thermal alone may confirm (renormalised)

# -------------------------------------------------------------- logging -----
LOG_DIR = "logs"
LOG_EVERY_N_STEPS = 8
VERBOSE = True

