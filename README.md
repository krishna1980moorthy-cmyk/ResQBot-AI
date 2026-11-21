# ResQBot AI

Autonomous Multi-Sensor Search-and-Rescue Rover

ResQBot AI is a Webots-based autonomous disaster-response
ground rover that integrates robotic mobility, AI-assisted
victim detection, obstacle avoidance and simulated
environmental hazard sensing.

## Technology
- Webots R2025a
- Python 3.11
- Fine-tuned YOLO11n
- LiDAR and proximity sensing
- GPS and orientation sensing
- 4WD skid-steer motion control
- Simulated gas sensing and thermal-camera proxy

## Project structure
- `controllers/resqbot_ai1/` - Main AI controller
- `controllers/resqbot_env_supervisor/` - Environment supervisor
- `protos/` - Rover definitions
- `worlds/` - Webots simulation environments

## How to run
1. Install Webots R2025a and Python 3.11.
2. Install Ultralytics, NumPy and OpenCV in the
   Python environment used by Webots.
3. Open `worlds/resqbot_ai_disaster_demo.wbt`.
4. Select the `resqbot_ai1` controller for the rover.
5. Start the Webots simulation.
6. Observe the live camera and telemetry output.

The trained YOLO model is provided in
`controllers/resqbot_ai1/resqbot_victim_best.pt`.

## Validation
Selected YOLO11n model results on a synthetic test set:
- Precision: 92.3%
- Recall: 80.0%
- mAP@50: 82.6%
- mAP@50-95: 36.7%

These results are from simulated imagery, not real
disaster-response testing.

## Limitations
Gas sensing and thermal sensing are simulated.
Victim localization, duplicate-alert handling and
complete hazard response require further verification.

## Future work
Physical prototype using Raspberry Pi 4 and ESP32,
with calibrated sensors and appropriate safety systems.
