# Webots Drone Controller: Target Find, Approach, and Land

## Overview

This Python script controls a simulated drone in the Webots environment. It uses a Finite State Machine (FSM) and computer vision (OpenCV for front camera, Webots Recognition for bottom camera) to perform the following task:
1.  **Takeoff:** Ascend to a predefined altitude.
2.  **Scan:** Rotate and search for a specific red target object (defined by `TARGET_CAR_NAME`) using the front camera and OpenCV color detection.
3.  **Approach:** Once the target is found and centered, move towards it using the front camera, slowing down as it gets closer.
4.  **Land:** When the target appears large enough and low enough in the front camera view, switch to the bottom camera (using Webots Recognition) and descend vertically onto the target, making fine position adjustments.

## Prerequisites

1.  **Webots:** A compatible version of the Webots simulator installed.
2.  **Python:** Python 3.x installed.
3.  **Libraries:**
    *   `controller`: Provided by Webots.
    *   `opencv-python`: Install using `pip install opencv-python`.
    *   `numpy`: Install using `pip install numpy`.
4.  **Webots World:**
    *   A Webots world file containing a drone model equipped with:
        *   GPS sensor named "drone_gps".
        *   Inertial Unit (IMU) sensor named "drone_imu".
        *   Gyro sensor named "gyro".
        *   A forward-facing Camera named "camera".
        *   A downward-facing Camera named "bottom_camera" **with a Recognition node attached**.
        *   Four motors named "front left propeller", "front right propeller", "rear left propeller", "rear right propeller". Motor rotation directions might need checking/adjusting in Webots or the script (`m2_speed`, `m3_speed` signs).
    *   A target object (e.g., a car model) with its `name` field set *exactly* to the value of `TARGET_CAR_NAME` in the script (default: `b"targetCar"`).
    *   The target object should have a distinct red color suitable for detection by the HSV ranges defined in the script.

## Configuration Parameters (Constants)

These parameters are defined at the beginning of the script and allow tuning the drone's behavior:

**Target Identification:**
*   `TARGET_CAR_NAME`: **Crucial.** Must match the `name` field of the target object in Webots (e.g., `b"myRedCar"`).

**Safety & Timeouts:**
*   `MAX_SCAN_TIME`, `MAX_APPROACH_TIME`, `MAX_LANDING_TIME`: Max duration (seconds) for each phase before aborting or reverting.
*   `LANDED_ALTITUDE_THRESHOLD`: Altitude (meters) below which the drone is considered landed.
*   `ALTITUDE_TOLERANCE`: Allowed error (meters) when reaching target altitudes.
*   `EMERGENCY_STOP_AREA_THRESHOLD_FRONT`: Relative area (0.0-1.0) in the front camera. If exceeded, triggers emergency stop (prevents collision).

**Control Gains (PID Tuning):**
*   `K_VERTICAL_THRUST`: Base thrust for hovering. Adjust if the drone struggles to take off or sinks/climbs too fast.
*   `K_VERTICAL_OFFSET`: Fine-tunes hover altitude.
*   `K_VERTICAL_P`: Aggressiveness of altitude corrections. Too high -> oscillation; too low -> slow response/drift.
*   `K_ROLL_P`, `K_PITCH_P`: Aggressiveness of stabilization. Too high -> oscillation; too low -> sluggish/unstable.

**Altitude Targets:**
*   `TARGET_ALTITUDE`: Cruising altitude (meters) for scanning/approach.
*   `LAND_HOLD_ALTITUDE`: Altitude (meters) to maintain if the target is lost during landing.

**Phase-Specific Control:**
*   **Scan:**
    *   `SCAN_YAW_GAIN`: How fast to turn towards the target during scan.
    *   `CENTERING_THRESHOLD`: Pixel tolerance for considering the target centered horizontally in the front camera.
    *   `SCAN_LOST_TARGET_YAW`: Speed of rotation when searching for a lost target.
*   **Approach:**
    *   `APPROACH_YAW_GAIN`: How fast to turn towards the target during approach.
    *   `APPROACH_TILT`: Base forward tilt (negative pitch). More negative = faster forward speed.
    *   `APPROACH_AREA_SPEED_FACTOR`: Controls how much the drone slows down based on target area (0.0-1.0). Higher means it slows down more significantly.
    *   `APPROACH_LOST_TARGET_FRAMES`: How many frames to wait before giving up on approach if target is lost.
*   **Landing Trigger (Front Cam):**
    *   `LANDING_TRIGGER_AREA_THRESHOLD`: Minimum *relative area* (0.0-1.0) required to initiate landing.
    *   `VERTICAL_POSITION_THRESHOLD`: Minimum *relative vertical position* (0.0-1.0, bottom=1.0) required to initiate landing. **Both must be met.**
*   **Landing (Bottom Cam):**
    *   `LANDING_CENTERING_P`: Aggressiveness of centering corrections during descent.
    *   `LANDING_MAX_CORRECTION_TILT`: Maximum allowed roll/pitch correction during landing.
    *   `LANDING_THRUST_BASE`: Base thrust during descent. Lower encourages descent.
    *   `LANDING_VERTICAL_P`: How much thrust changes with altitude during descent.
    *   `LAND_LOST_TARGET_FRAMES_TIMEOUT`: How many frames to wait before aborting landing if target is lost.

**Vision (Front Camera - OpenCV):**
*   `MIN_TARGET_SIZE`: Minimum pixel area to consider a detection valid.
*   `RED_HSV_RANGES`: **Crucial.** List of HSV tuples `((H_low, S_low, V_low), (H_high, S_high, V_high))` defining the target color. Use tools like GIMP or online HSV pickers, adapting to the Webots lighting.

## Code Structure

*   **Imports & Constants:** Setup libraries and tuning parameters.
*   **`DroneState` Class:** Manages the drone's current phase and status.
*   **Robot Initialization:** Gets handles to sensors, cameras, and motors. Enables them.
*   **Helper Functions:** `clamp`, `emergency_stop`, `land_complete`.
*   **Vision Functions:**
    *   `detect_red_target_opencv`: Uses OpenCV (HSV filtering, contours) on the front camera image.
    *   `check_recognition_bottom`: Uses Webots' built-in Recognition feature on the bottom camera image.
*   **`display_camera_feed`:** Shows camera output with debug overlays using OpenCV.
*   **Main Loop:**
    *   Reads sensor data.
    *   Performs safety checks.
    *   Manages state transitions and timeouts.
    *   Calculates stabilization and altitude control signals (PID-like).
    *   Selects the active camera based on the current phase.
    *   Processes the image from the active camera.
    *   Executes phase-specific logic (Takeoff, Scan, Approach, Land) modifying control inputs.
    *   Calculates final motor velocities (mixing).
    *   Sends commands to motors.
    *   Handles OpenCV display and exit keys.
*   **Cleanup:** Stops motors and closes windows on exit.

## Workflow / Finite State Machine (FSM)

The drone operates based on the `state.phase` variable:

1.  **`takeoff`**:
    *   Goal: Reach `TARGET_ALTITUDE`.
    *   Action: Apply vertical thrust to climb. Stabilize roll/pitch.
    *   Transition: `altitude >= TARGET_ALTITUDE - ALTITUDE_TOLERANCE` -> **`scan`**.
2.  **`scan`**: (Front Camera - OpenCV)
    *   Goal: Find and center the red target.
    *   Action: Maintain altitude. If target lost, yaw slowly (`SCAN_LOST_TARGET_YAW`). If target found but off-center, yaw towards it (`SCAN_YAW_GAIN`).
    *   Transition: Target found AND `abs(offset_x) < CENTERING_THRESHOLD` -> **`approach`**.
    *   Timeout: `MAX_SCAN_TIME` exceeded -> **`scan`** (restarts scan).
3.  **`approach`**: (Front Camera - OpenCV)
    *   Goal: Move towards the centered target, slowing down. Trigger landing.
    *   Action: Maintain altitude. Apply forward pitch (`APPROACH_TILT`, scaled by `APPROACH_AREA_SPEED_FACTOR`). Apply yaw correction (`APPROACH_YAW_GAIN`).
    *   Transition: Target `relative_area > LANDING_TRIGGER_AREA_THRESHOLD` AND Target `cy / cam_h > VERTICAL_POSITION_THRESHOLD` AND Bottom Cam OK -> **`land`**.
    *   Abort: Target lost for `APPROACH_LOST_TARGET_FRAMES` -> **`scan`**.
    *   Timeout: `MAX_APPROACH_TIME` exceeded -> **`scan`**.
    *   Safety: `relative_area > EMERGENCY_STOP_AREA_THRESHOLD_FRONT` -> Emergency Stop.
4.  **`land`**: (Bottom Camera - Recognition)
    *   Goal: Descend vertically onto the target detected by the bottom camera.
    *   Action: Apply reduced vertical thrust (`LANDING_THRUST_BASE`, `LANDING_VERTICAL_P`) to descend. Apply roll/pitch corrections (`LANDING_CENTERING_P`) to stay centered over the target. No yaw.
    *   Transition: `altitude < LANDED_ALTITUDE_THRESHOLD` -> **Landed (Emergency Stop)**.
    *   Abort: Target lost for `LAND_LOST_TARGET_FRAMES_TIMEOUT` -> **`scan`**.
    *   Timeout: `MAX_LANDING_TIME` exceeded -> Emergency Stop.
    *   Hold: If target lost during landing, maintain `LAND_HOLD_ALTITUDE` while waiting for timeout or re-acquisition.

## Key Functions

*   `detect_red_target_opencv(...)`: Heart of front-camera vision. Finds the largest red contour.
*   `check_recognition_bottom(...)`: Uses Webots API to identify the pre-defined target below the drone.
*   `display_camera_feed(...)`: Visual feedback showing what the active camera sees and detection results.
*   `emergency_stop(...)`: Halts all motors and ends the script execution safely.

## Running the Script

1.  Open the corresponding Webots world file.
2.  Ensure the drone in the world has a Robot controller set to `<extern>`.
3.  Run this Python script from a terminal *while the simulation is running*.
4.  An OpenCV window titled "Camera View" should appear, showing the drone's camera feed and status information.
5.  Press 'q' or 'Esc' in the OpenCV window to stop the script and the drone.

## Troubleshooting / Tuning Tips

*   **Target Not Found (Scan):**
    *   Check `TARGET_CAR_NAME` *exactly* matches the object name in Webots (case-sensitive).
    *   Adjust `RED_HSV_RANGES` significantly. Use an HSV color picker on a screenshot from Webots under the simulation lighting. Check Saturation and Value minimums (too high might ignore shadows).
    *   Lower `MIN_TARGET_SIZE` if the target is initially very small.
    *   Ensure the target is actually visible from the drone's starting position/altitude.
*   **Unstable Flight (Oscillations):**
    *   Reduce `K_ROLL_P`, `K_PITCH_P` gains.
    *   Reduce `K_VERTICAL_P` gain if altitude oscillates.
*   **Sluggish Flight / Drifting:**
    *   Increase `K_ROLL_P`, `K_PITCH_P` gains cautiously.
    *   Increase `K_VERTICAL_P` if altitude control is poor.
    *   Adjust `K_VERTICAL_THRUST` and `K_VERTICAL_OFFSET` for better hovering.
*   **Approach Issues:**
    *   Too fast/overshoots: Decrease `APPROACH_TILT` (make less negative), increase `APPROACH_AREA_SPEED_FACTOR`.
    *   Too slow: Increase `APPROACH_TILT` (make more negative), decrease `APPROACH_AREA_SPEED_FACTOR`.
    *   Wobbly approach: Decrease `APPROACH_YAW_GAIN`.
    *   Doesn't trigger landing: Check `LANDING_TRIGGER_AREA_THRESHOLD` and `VERTICAL_POSITION_THRESHOLD`. Are they being met? Observe values printed in the console. Check if the bottom camera has a Recognition node working.
*   **Landing Issues:**
    *   Doesn't descend: Decrease `LANDING_THRUST_BASE`, potentially increase `LANDING_VERTICAL_P`. Ensure `altitude` is decreasing as expected.
    *   Descends too fast: Increase `LANDING_THRUST_BASE`.
    *   Poor centering: Adjust `LANDING_CENTERING_P`. Check `LANDING_MAX_CORRECTION_TILT`. Ensure bottom camera Recognition node is accurately reporting position. Check pitch correction sign convention (`-LANDING_CENTERING_P * offset_y`).
    *   Loses target during landing: Increase `LAND_LOST_TARGET_FRAMES_TIMEOUT`. Ensure Recognition works reliably at close range.
*   **Motor Issues:**
    *   Drone flips on takeoff: Check motor names and the signs used in the motor mixing calculation (`m1_speed` to `m4_speed`). Especially check if `m2_speed` and `m3_speed` need their signs flipped (`-m2_speed`, `-m3_speed`).

## Limitations

*   No explicit obstacle avoidance (relies on timeouts and front area check).
*   Assumes a relatively flat landing surface.
*   Relies heavily on accurate color detection (OpenCV) and object recognition (Webots). Performance may vary with lighting conditions and target appearance.
*   PID gains likely need tuning for different drone models or physics settings.