from controller import Robot, GPS, InertialUnit, Gyro, Motor, Camera
import cv2
import numpy as np
import math
import time

TIME_STEP = 32

# Safety parameters
MAX_SCAN_TIME = 30      # seconds
MAX_APPROACH_TIME = 20  # seconds
MAX_LANDING_TIME = 25   # seconds for landing phase
EMERGENCY_STOP_AREA_THRESHOLD_FRONT = 0.5 # % of front camera view area
LANDING_APPROACH_AREA_THRESHOLD = 0.25 # % Area in front cam to trigger landing - CHECK THIS VALUE
LANDED_ALTITUDE_THRESHOLD = 0.15 # Altitude considered 'landed'

# Control gains
K_VERTICAL_THRUST = 68.0      # Base thrust
K_VERTICAL_OFFSET = 0.6       # Offset for altitude PID
K_VERTICAL_P      = 3.0       # Proportional gain for altitude
K_ROLL_P          = 15.0      # Proportional gain for roll stabilization
K_PITCH_P         = 15.0      # Proportional gain for pitch stabilization

# Altitude
TARGET_ALTITUDE = 3.0

# --- Phase Specific Gains ---
# Scan
SCAN_YAW_GAIN      = 0.3
CENTERING_THRESHOLD = 50
SCAN_LOST_TARGET_YAW = 0.15

# Approach
APPROACH_YAW_GAIN  = 0.05
APPROACH_TILT      = -0.15
APPROACH_AREA_SPEED_FACTOR = 0.8
APPROACH_LOST_TARGET_FRAMES = 15

# Landing - **NEEDS TUNING**
LANDING_VERTICAL_P = 15.0
LANDING_THRUST_BASE = 60.0 # Try adjusting this first if descent is too fast/slow
LANDING_CENTERING_P = 0.005 # Try adjusting this if centering is wrong
LANDING_MAX_CORRECTION_TILT = 0.05

# Minimum contour area to consider a valid target
MIN_TARGET_SIZE = 50

# Color detection: wide red ranges
RED_HSV_RANGES = [
    ((0,   70,  70), (10,  255, 255)),
    ((165, 70,  70), (180, 255, 255))
]

# State tracking (same as before)
class DroneState:
    def __init__(self):
        self.phase = "takeoff"
        self.target_found_front = False
        self.target_found_bottom = False
        self.target_lost_frames = 0
        self.phase_start_time = time.time()
        self.emergency_stop = False
        self.active_camera_name = "camera"

    def reset_phase_timer(self):
        self.phase_start_time = time.time()

    def get_phase_duration(self):
        return time.time() - self.phase_start_time

    def should_abort_phase(self):
        duration = self.get_phase_duration()
        if self.phase == "scan" and duration > MAX_SCAN_TIME:
            print(f"⚠️ Scan phase timeout ({duration:.1f}s > {MAX_SCAN_TIME}s)")
            return True
        if self.phase == "approach" and duration > MAX_APPROACH_TIME:
            print(f"⚠️ Approach phase timeout ({duration:.1f}s > {MAX_APPROACH_TIME}s)")
            return True
        if self.phase == "land" and duration > MAX_LANDING_TIME:
            print(f"⚠️ Landing phase timeout ({duration:.1f}s > {MAX_LANDING_TIME}s)")
            emergency_stop("Landing Timeout")
            return True
        return False

robot = Robot()

# --- Initialize Devices ---
gps = robot.getDevice("drone_gps")
imu = robot.getDevice("drone_imu")
gyro = robot.getDevice("gyro")
front_camera = robot.getDevice("camera")

# *** CHECK CONSOLE FOR THIS OUTPUT ***
bottom_camera = None # Initialize as None
try:
    bottom_camera_device = robot.getDevice("bottom_camera") # Check name carefully!
    if bottom_camera_device:
        bottom_camera = bottom_camera_device # Assign only if found
        bottom_camera.enable(TIME_STEP)
        print("✅ Bottom camera initialized successfully.")
    else:
        # This case might happen if getDevice doesn't raise error but returns null/invalid
         print("❌ Bottom camera device found by name, but seems invalid.")
except Exception as e:
    # This catches cases where the device name doesn't exist
    print(f"❌ Error initializing bottom camera (getDevice failed): {e}. Landing will not be possible.")
    # bottom_camera remains None


gps.enable(TIME_STEP)
imu.enable(TIME_STEP)
gyro.enable(TIME_STEP)
front_camera.enable(TIME_STEP)

motor_names = ["front left propeller", "front right propeller",
               "rear left propeller", "rear right propeller"]
motors = [robot.getDevice(name) for name in motor_names]
for m in motors:
    m.setPosition(float("inf"))
    m.setVelocity(0.0)

state = DroneState()

# --- OpenCV Windows ---
cv2.namedWindow("Camera View", cv2.WINDOW_NORMAL)

# --- Helper Functions (clamp, emergency_stop, land_complete, detect_red_target, process_camera_image) ---
# (Keep the helper functions exactly as in the previous response)
# ... (Insert the helper functions here) ...
def clamp(value, min_val, max_val):
    """Clamps a value between min_val and max_val."""
    return max(min_val, min(value, max_val))

def emergency_stop(reason=""):
    """Stops all motors immediately."""
    if not state.emergency_stop: # Prevent multiple prints
        print(f"⚠️ EMERGENCY STOP ACTIVATED! Reason: {reason}")
        state.emergency_stop = True
    for m in motors:
        m.setVelocity(0.0)

def land_complete():
    """Action when landing is finished."""
    print("✅ Landing complete. Stopping motors.")
    emergency_stop("Landed") # Use emergency stop to halt motors

def detect_red_target(image_bgr, min_area):
    """
    Detects the largest red contour in the image.
    Returns (found, cx, cy, area, bbox)
    """
    try:
        if image_bgr is None: # Add check if image is valid
            print("Error: detect_red_target received None image.")
            return False, 0, 0, 0, None

        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for (lower, upper) in RED_HSV_RANGES:
            lower_np = np.array(lower)
            upper_np = np.array(upper)
            current_mask = cv2.inRange(hsv, lower_np, upper_np)
            mask |= current_mask

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return False, 0, 0, 0, None

        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        if area > min_area:
            x, y, w, h = cv2.boundingRect(largest_contour)
            cx = x + w // 2
            cy = y + h // 2
            return True, cx, cy, area, (x, y, x + w, y + h)
        else:
            return False, 0, 0, area, None # Report area even if too small

    except cv2.error as e:
         print(f"OpenCV Error in detect_red_target: {e}")
         return False, 0, 0, 0, None
    except Exception as e:
        print(f"Error in detect_red_target: {e}")
        return False, 0, 0, 0, None

def process_camera_image(camera_device):
    """Gets image, converts to BGR, detects target, and displays."""
    if camera_device is None: # Check if camera device is valid
        # print("Error: process_camera_image received None device.") # Reduce noise
        return None, None, False, 0, 0, 0, None

    img_data = camera_device.getImage()
    if not img_data:
        # print(f"Warning: No image data from camera {camera_device.getName()}") # Reduce noise
        return None, None, False, 0, 0, 0, None # img, debug_img, found, cx, cy, area, bbox

    w = camera_device.getWidth()
    h = camera_device.getHeight()
    # Check if image dimensions are valid
    if w <= 0 or h <= 0:
        print(f"Warning: Invalid dimensions ({w}x{h}) for camera {camera_device.getName()}")
        return None, None, False, 0, 0, 0, None

    try:
        img_array = np.frombuffer(img_data, np.uint8).reshape((h, w, 4))
        bgr_img = cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)
        debug_img = bgr_img.copy() # Create copy for drawing

        # Call detection function (ensure it handles potential errors)
        found, cx, cy, area, bbox = detect_red_target(bgr_img, MIN_TARGET_SIZE)

        # Draw visualization on the debug image
        if found and bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.circle(debug_img, (cx, cy), 5, (255, 0, 0), -1)
            cv2.putText(debug_img, f"Area: {area:.0f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        # Draw center crosshair
        cv2.line(debug_img, (w // 2, h // 2 - 10), (w // 2, h // 2 + 10), (0, 255, 255), 1)
        cv2.line(debug_img, (w // 2 - 10, h // 2), (w // 2 + 10, h // 2), (0, 255, 255), 1)
        # Add phase text
        cv2.putText(debug_img, f"Phase: {state.phase}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        cv2.putText(debug_img, f"Cam: {state.active_camera_name}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)


        # Display the debug image
        cv2.imshow("Camera View", debug_img)

        return bgr_img, debug_img, found, cx, cy, area, bbox

    except Exception as e:
        print(f"Error processing/displaying image from {camera_device.getName()}: {e}")
        # Try to display a blank image or skip display if it fails
        cv2.imshow("Camera View", np.zeros((h,w,3), dtype=np.uint8)) # Show black screen on error
        return None, None, False, 0, 0, 0, None
# --- Main Loop ---
print("Starting drone control loop...")
while robot.step(TIME_STEP) != -1 and not state.emergency_stop:
    try:
        # --- Read Sensors (same as before) ---
        roll, pitch, yaw = imu.getRollPitchYaw()
        altitude = gps.getValues()[2]
        roll_rate, pitch_rate, yaw_rate = gyro.getValues()

        # --- Safety Checks (same as before) ---
        if abs(roll) > 0.8 or abs(pitch) > 0.8:
            emergency_stop("Extreme angles")
            continue
        if altitude < -0.1 :
             emergency_stop("Drone below ground")
             continue
        if not state.emergency_stop and state.should_abort_phase():
            if state.phase != "land":
                print(f"Phase '{state.phase}' timed out. Returning to SCAN.")
                state.phase = "scan"
                state.reset_phase_timer()
                state.target_lost_frames = 0

        # --- Common PID Calculations (same as before) ---
        clamped_diff_alt = clamp(TARGET_ALTITUDE - altitude + K_VERTICAL_OFFSET, -1.0, 1.0)
        vertical_input_pid = K_VERTICAL_P * (clamped_diff_alt ** 3)
        roll_clamped  = clamp(roll,  -0.5,  0.5)
        pitch_clamped = clamp(pitch, -0.5,  0.5)
        roll_stabilization  = K_ROLL_P  * roll_clamped  + roll_rate
        pitch_stabilization = K_PITCH_P * pitch_clamped + pitch_rate

        # --- Default Inputs (same as before) ---
        vertical_thrust = K_VERTICAL_THRUST + vertical_input_pid
        roll_input      = roll_stabilization
        pitch_input     = pitch_stabilization
        yaw_input       = 0.0

        # --- Finite State Machine (FSM) ---
        # *** DEBUG: Choose camera based on state ***
        current_camera = None
        previous_active_cam = state.active_camera_name # Store previous name
        if state.phase == "land" and bottom_camera:
            current_camera = bottom_camera
            state.active_camera_name = "bottom_camera"
        else:
            # Default to front camera if not landing or if bottom camera failed init
            current_camera = front_camera
            state.active_camera_name = "camera" # Name matches front cam name

        # *** DEBUG: Print if camera changed ***
        if state.active_camera_name != previous_active_cam:
            print(f"DEBUG: Switched active camera to -> {state.active_camera_name}")

        # Process image from the active camera
        # Make sure current_camera is valid before processing
        if current_camera:
            bgr_img, debug_img, found, cx, cy, area, bbox = process_camera_image(current_camera)
            if bgr_img is not None: # Check if processing was successful
                 cam_w = current_camera.getWidth()
                 cam_h = current_camera.getHeight()
                 # Update target found status based on active camera
                 if state.active_camera_name == "camera":
                     state.target_found_front = found
                 elif state.active_camera_name == "bottom_camera":
                     state.target_found_bottom = found
            else:
                 # Image processing failed, reset found status for the active camera
                 found = False # Ensure 'found' is False if image processing failed
                 if state.active_camera_name == "camera": state.target_found_front = False
                 elif state.active_camera_name == "bottom_camera": state.target_found_bottom = False
                 # Assign default values if processing failed
                 cx, cy, area, bbox = 0, 0, 0, None
                 cam_w, cam_h = (128, 128) # Assign default dimensions to avoid errors later if needed

        else:
            # This case should ideally not happen if front_camera is always initialized
            print("CRITICAL WARNING: No valid camera selected.")
            found = False
            state.target_found_front = False
            state.target_found_bottom = False
            cx, cy, area, bbox = 0, 0, 0, None
            cam_w, cam_h = (128, 128) # Assign default dimensions


        # --- Phase Logic ---
        if state.phase == "takeoff":
            # (Same as before)
            print(f"[TAKEOFF] Altitude: {altitude:.2f} / {TARGET_ALTITUDE:.2f}")
            if altitude >= TARGET_ALTITUDE - 0.1:
                print("🟢 Altitude reached. Switching to SCAN phase.")
                state.phase = "scan"
                state.reset_phase_timer()

        elif state.phase == "scan":
            # (Same as before - uses state.target_found_front implicitly via camera selection)
            if state.target_found_front:
                state.target_lost_frames = 0
                offset_x = cx - (cam_w // 2)
                if abs(offset_x) < CENTERING_THRESHOLD:
                    print("🎯 Target centered in SCAN -> APPROACH phase")
                    state.phase = "approach"
                    state.reset_phase_timer()
                else:
                    yaw_input = -SCAN_YAW_GAIN * (offset_x / (cam_w // 2))
                    yaw_input = clamp(yaw_input, -0.5, 0.5)
                    # print(f"🔄 Scanning: Target found, centering. Offset={offset_x}, Yaw={yaw_input:.2f}") # Reduce print frequency
            else:
                state.target_lost_frames += 1
                if state.target_lost_frames > 5:
                    yaw_input = SCAN_LOST_TARGET_YAW
                    if state.target_lost_frames % 10 == 1: # Print less often
                         print(f"🔍 Scanning: Target lost ({state.target_lost_frames} frames), rotating. Yaw={yaw_input:.2f}")


        elif state.phase == "approach":
            if state.target_found_front:
                state.target_lost_frames = 0
                offset_x = cx - (cam_w // 2)
                relative_area = area / (cam_w * cam_h) if (cam_w * cam_h) > 0 else 0

                # *** DEBUG: Print area for transition check ***
                if state.target_lost_frames == 0: # Print only when target is found
                    print(f"[APPROACH] Target Area: {relative_area:.3f} (Threshold: {LANDING_APPROACH_AREA_THRESHOLD})", end=" ")

                # Check for Landing Condition
                if relative_area > LANDING_APPROACH_AREA_THRESHOLD:
                    print(" -> Threshold MET!") # Add confirmation
                    if bottom_camera:
                        print(f"✅ Switching to LAND phase.")
                        state.phase = "land"
                        state.reset_phase_timer()
                        continue # Skip rest of approach logic for this step
                    else:
                        print("⚠️ Target close but no bottom camera! Holding position.")
                        pitch_input = pitch_stabilization
                        yaw_input = 0
                # Check for Emergency Stop Condition
                elif relative_area > EMERGENCY_STOP_AREA_THRESHOLD_FRONT:
                     print(" -> EMERGENCY STOP THRESHOLD MET!")
                     emergency_stop(f"Target too close in front view ({relative_area:.2%})")
                     continue

                else:
                    # Approach Control (Same as before)
                    print(" -> Approaching...") # Continue approach log on same line
                    area_factor = 1.0 - min(relative_area / LANDING_APPROACH_AREA_THRESHOLD, APPROACH_AREA_SPEED_FACTOR)
                    forward_tilt = APPROACH_TILT * area_factor
                    pitch_input = pitch_stabilization + K_PITCH_P * clamp(forward_tilt, -0.3, 0.0)
                    yaw_input = -APPROACH_YAW_GAIN * (offset_x / (cam_w // 2))
                    yaw_input = clamp(yaw_input, -0.3, 0.3)
                    # Combined log from above print statements

            else:
                # Target lost during approach
                state.target_lost_frames += 1
                if state.target_lost_frames > APPROACH_LOST_TARGET_FRAMES:
                    print(f"\n⚠️ Target lost in APPROACH ({state.target_lost_frames} frames) -> returning to SCAN")
                    state.phase = "scan"
                    state.reset_phase_timer()
                else:
                    # Optionally add a print if target just lost but not yet switching back
                    if state.target_lost_frames == 1: print("\n[APPROACH] Target lost...")


        elif state.phase == "land":
            # This phase now uses state.target_found_bottom implicitly via camera selection
            if not bottom_camera: # Should have been caught by approach phase, but double check
                print("❌ CRITICAL: In LAND phase but no bottom camera! Returning to SCAN.")
                state.phase = "scan"
                state.reset_phase_timer()
                continue

            # Landing Vertical Control (Same as before)
            vertical_thrust = LANDING_THRUST_BASE + LANDING_VERTICAL_P * altitude
            vertical_thrust = clamp(vertical_thrust, 0, K_VERTICAL_THRUST * 1.1)

            # Landing Horizontal Control (Centering)
            roll_correction = 0.0
            pitch_correction = 0.0
            log_msg = f"[LAND] Alt: {altitude:.2f} Thrust: {vertical_thrust:.1f} " # Start log message

            if state.target_found_bottom:
                state.target_lost_frames = 0
                offset_x = cx - (cam_w // 2)
                offset_y = cy - (cam_h // 2)
                roll_correction = LANDING_CENTERING_P * offset_x
                pitch_correction = -LANDING_CENTERING_P * offset_y
                roll_correction = clamp(roll_correction, -LANDING_MAX_CORRECTION_TILT, LANDING_MAX_CORRECTION_TILT)
                pitch_correction = clamp(pitch_correction, -LANDING_MAX_CORRECTION_TILT, LANDING_MAX_CORRECTION_TILT)

                log_msg += f"Target: Yes Offs({offset_x},{offset_y}) Corr({roll_correction:.3f},{pitch_correction:.3f})"

                roll_input = roll_stabilization + K_ROLL_P * roll_correction
                pitch_input = pitch_stabilization + K_PITCH_P * pitch_correction
                yaw_input = 0

            else:
                # Target lost during landing
                state.target_lost_frames += 1
                log_msg += f"Target: No ({state.target_lost_frames} frames) Descending blind"
                roll_input = roll_stabilization
                pitch_input = pitch_stabilization
                yaw_input = 0

            print(log_msg) # Print the combined log message

            # Check if Landed
            if altitude < LANDED_ALTITUDE_THRESHOLD:
                land_complete()
                continue

        # --- Final Motor Commands (Same as before) ---
        front_left_motor_input  = vertical_thrust - roll_input + pitch_input - yaw_input
        front_right_motor_input = vertical_thrust + roll_input + pitch_input + yaw_input
        rear_left_motor_input   = vertical_thrust - roll_input - pitch_input + yaw_input
        rear_right_motor_input  = vertical_thrust + roll_input - pitch_input - yaw_input

        motors[0].setVelocity( front_left_motor_input)
        motors[1].setVelocity(-front_right_motor_input)
        motors[2].setVelocity(-rear_left_motor_input)
        motors[3].setVelocity( rear_right_motor_input)

        # --- OpenCV Window Update ---
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("User pressed 'q'. Exiting.")
            break

    except Exception as e:
        print(f"\n❌❌❌ An error occurred in the main loop: {e} ❌❌❌")
        import traceback
        traceback.print_exc()
        emergency_stop("Main loop exception")
        break

# --- Cleanup ---
print("Simulation ended or stopped.")
if not state.emergency_stop: # Ensure motors are stopped if loop exited normally
    emergency_stop("End of script")
cv2.destroyAllWindows()