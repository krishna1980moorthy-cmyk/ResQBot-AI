# ResQBot AI — Webots Main Controller (Stage 1)

```
controllers/resqbot_ai/
├── resqbot_ai.py      main controller — sense → fuse → navigate → mission → motors → log
├── config.py          ALL device names, thresholds, gas zones (edit this first)
├── motor_control.py   4WD skid-steer, soft accel ramp, emergency stop
├── navigation.py      LiDAR sectors, avoidance, stuck detect + recovery, victim projection
├── rgb_detection.py   RGB camera + YOLO11n person detection (auto-disables if missing)
├── thermal_sensor.py  32×24 simulated thermal (camera-proxy or synthetic ambient)
├── gas_sensor.py      Gaussian-plume gas zones → ppm, SAFE/WARN/DANGER
├── sensor_fusion.py   RGB+thermal late fusion w/ spatial agreement, N-frame confirm
├── mission_logic.py   SEARCH / INSPECT / ALERT / HAZARD / STOP state machine
└── metrics.py         logs/run_*.csv + logs/summary_*.json (real measured values only)
```

## Setup
1. Copy the folder into your Webots project `controllers/`. In the rover node set `controller "resqbot_ai"`.
2. Edit `config.DEVICE_NAMES` to match your PROTO (4 motors, `lidar`, `rgb_camera`, `thermal_camera`, `gps`, `inertial_unit`).
3. Wheels spin the wrong way → flip `MOTOR_DIRECTION`.
4. YOLO: `pip install ultralytics numpy` into the Python Webots uses (Preferences → Python command). `yolo11n.pt` downloads on first run, or put it in this folder.
5. Gas zones: set `GAS_ZONES` (world x, y, radius, peak ppm) to match hazard spots in your world.

## Thermal simulation
Add a second low-res Camera named `thermal_camera` (e.g. 64×48) next to the RGB camera. Its brightness maps to °C, so give victim models a bright/emissive material. Without it, thermal sends ambient noise only and fusion uses RGB alone.

## Fusion rules
- `conf = 0.55·RGB + 0.45·thermal(aligned)`, plus a bonus when both agree. Thermal only counts if the hot region lines up with the YOLO box.
- Suspect ≥ 0.40 → INSPECT (slow down, centre on target). Confirmed at ≥ 0.70 for 3 frames and ≥ 1 s dwell → ALERT.
- If YOLO isn't available, thermal alone can confirm (`THERMAL_ONLY_FALLBACK`). Set it False for strict mode.
- Victim world position = pose + LiDAR range along the bearing, so the same victim isn't logged twice within 1.5 m.
- Gas DANGER overrides SEARCH/INSPECT → back out and turn away.
- A safety layer in the main loop blocks forward motion closer than 0.21 m, whatever the state.

## Stage 3 mapping (physical rover)
| Stage 1 module | Stage 3 home |
|---|---|
| resqbot_ai, navigation, rgb_detection, thermal_sensor, sensor_fusion, mission_logic, metrics | Raspberry Pi 4 |
| motor_control (`set_tracks`/`stop`) | ESP32 over UART, e.g. `$M,<left>,<right>*CS\n` + watchdog stop |
| gas_sensor (simulated) | ESP32 ADC (MQ-2/MQ-135) → UART to Pi |
| thermal_sensor (camera proxy) | MLX90640 over I²C (Pi) |

Only `MotorController` and the sensor `read()` methods change. The mission and fusion code stays the same.
