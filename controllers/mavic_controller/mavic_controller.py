from controller import Robot, GPS, InertialUnit, Gyro, Motor, Camera
import cv2
import numpy as np
import math
import time

# --- Constants ---
TIME_STEP = 32

# Safety parameters
MAX_SCAN_TIME = 30
MAX_APPROACH_TIME = 30
MAX_HOVER_LOW_TIME = 20
MAX_LANDING_TIME = 35 # Increased slightly as it might hover waiting for detection
EMERGENCY_STOP_AREA_THRESHOLD_FRONT = 0.5
LANDED_ALTITUDE_THRESHOLD = 0.15
ALTITUDE_TOLERANCE = 0.1

# Control gains
K_VERTICAL_THRUST = 68.0
K_VERTICAL_OFFSET = 0.6
K_VERTICAL_P      = 3.0
K_ROLL_P          = 15.0
K_PITCH_P         = 15.0

# Altitude Targets
TARGET_ALTITUDE = 3.0
HOVER_LOW_ALTITUDE = 1.0

# --- Phase Specific Gains / Thresholds ---
# Scan
SCAN_YAW_GAIN      = 0.3; CENTERING_THRESHOLD = 50; SCAN_LOST_TARGET_YAW = 0.15
# Approach
APPROACH_YAW_GAIN  = 0.05; APPROACH_TILT = -0.15; APPROACH_AREA_SPEED_FACTOR = 0.8
APPROACH_LOST_TARGET_FRAMES = 15; OVERHEAD_CENTERING_THRESHOLD = 25; OVERHEAD_PROXIMITY_AREA = 0.03
# Hover Low
HOVER_LOW_CENTERING_P = 0.01; HOVER_LOW_MAX_CORRECTION = 0.1;
# Landing
LANDING_THRUST_BASE = 55.0; LANDING_VERTICAL_P = 8.0; LANDING_CENTERING_P = 0.005
LANDING_MAX_CORRECTION_TILT = 0.05; LAND_LOST_TARGET_FRAMES_TIMEOUT = 60 # Frames (~2s) to wait in land phase before reverting if target lost

# Vision parameters
MIN_TARGET_SIZE = 50
RED_HSV_RANGES = [((0, 70, 70), (10, 255, 255)), ((165, 70, 70), (180, 255, 255))]

# --- State Tracking Class ---
class DroneState: # (No changes from previous version)
    def __init__(self):
        self.phase = "takeoff"; self.target_found_front = False; self.target_found_bottom = False
        self.target_lost_frames = 0; self.phase_start_time = time.time(); self.emergency_stop = False
        self.active_camera_name = "camera"
    def reset_phase_timer(self): self.phase_start_time = time.time()
    def get_phase_duration(self): return time.time() - self.phase_start_time
    def should_abort_phase(self):
        duration = self.get_phase_duration()
        phase_timeouts = {"scan": MAX_SCAN_TIME, "approach": MAX_APPROACH_TIME, "hover_low": MAX_HOVER_LOW_TIME, "land": MAX_LANDING_TIME}
        if self.phase in phase_timeouts and duration > phase_timeouts[self.phase]:
            print(f"⚠️ {self.phase.upper()} phase timeout ({duration:.1f}s > {phase_timeouts[self.phase]}s)")
            if self.phase == "land": emergency_stop("Landing Timeout"); return True
            else: return True # Signal other timeouts
        return False

# --- Robot Initialization ---
# (No changes from previous version - ensure bottom_camera name is correct in your Webots world)
robot = Robot(); state = DroneState()
gps = robot.getDevice("drone_gps"); imu = robot.getDevice("drone_imu"); gyro = robot.getDevice("gyro")
front_camera = robot.getDevice("camera")
bottom_camera = None
try:
    bottom_camera_device = robot.getDevice("bottom_camera")
    if bottom_camera_device: bottom_camera = bottom_camera_device; bottom_camera.enable(TIME_STEP); print("✅ Bottom camera initialized.")
    else: print("❌ Bottom camera device seems invalid.")
except Exception as e: print(f"❌ Error initializing bottom camera: {e}.")
gps.enable(TIME_STEP); imu.enable(TIME_STEP); gyro.enable(TIME_STEP); front_camera.enable(TIME_STEP)
motor_names = ["front left propeller", "front right propeller", "rear left propeller", "rear right propeller"]
motors = [robot.getDevice(name) for name in motor_names];
for m in motors: m.setPosition(float("inf")); m.setVelocity(0.0)

# --- OpenCV Windows ---
cv2.namedWindow("Camera View", cv2.WINDOW_NORMAL)
# cv2.namedWindow("Bottom Cam Mask", cv2.WINDOW_NORMAL) # Optional debug window

# --- Helper Functions ---
# (No changes from previous version for clamp, emergency_stop, land_complete, detect_red_target, process_camera_image)
def clamp(value, min_val, max_val): return max(min_val, min(value, max_val))
def emergency_stop(reason=""):
    global state;
    if not state.emergency_stop: print(f"⚠️ EMERGENCY STOP: {reason}"); state.emergency_stop = True
    for m in motors: m.setVelocity(0.0)
def land_complete(): print("✅ Landing complete."); emergency_stop("Landed")
def detect_red_target(image_bgr, min_area, active_camera_name): # Pass name for debug
    try: # Simplified for brevity - keep full error handling from previous code
        if image_bgr is None: return False, 0, 0, 0, None
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for (lower, upper) in RED_HSV_RANGES: mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))
        # --- Optional Conditional Mask Display ---
        # if active_camera_name == (bottom_camera.getName() if bottom_camera else "none"): cv2.imshow("Bottom Cam Mask", mask)
        # else: if cv2.getWindowProperty("Bottom Cam Mask", 0) >= 0: cv2.destroyWindow("Bottom Cam Mask")
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return False, 0, 0, 0, None
        largest_contour = max(contours, key=cv2.contourArea); area = cv2.contourArea(largest_contour)
        if area > min_area: x, y, w, h = cv2.boundingRect(largest_contour); return True, x + w // 2, y + h // 2, area, (x, y, x + w, y + h)
        else: return False, 0, 0, area, None
    except Exception as e: print(f"Detect Error: {e}"); return False, 0, 0, 0, None
def process_camera_image(camera_device, active_camera_name): # Pass name
    global state;
    if camera_device is None: return None, None, False, 0, 0, 0, None
    img_data = camera_device.getImage(); w = camera_device.getWidth(); h = camera_device.getHeight()
    if not img_data or w <= 0 or h <= 0: return None, None, False, 0, 0, 0, None
    try: # Simplified - keep full error handling/drawing from previous
        bgr_img = cv2.cvtColor(np.frombuffer(img_data, np.uint8).reshape((h, w, 4)), cv2.COLOR_BGRA2BGR)
        debug_img = bgr_img.copy()
        found, cx, cy, area, bbox = detect_red_target(bgr_img, MIN_TARGET_SIZE, active_camera_name) # Pass name
        # Drawing code omitted for brevity - keep from previous version
        if found and bbox: x1, y1, x2, y2 = bbox; cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0,0,255), 1); cv2.circle(debug_img, (cx,cy), 3, (255,0,0), -1)
        cv2.putText(debug_img, f"P:{state.phase} C:{active_camera_name}", (5,15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)
        cv2.imshow("Camera View", debug_img)
        return bgr_img, debug_img, found, cx, cy, area, bbox
    except Exception as e: print(f"Process Img Err: {e}"); return None, None, False, 0, 0, 0, None

# --- Main Control Loop ---
print("Starting drone control loop...")
while robot.step(TIME_STEP) != -1 and not state.emergency_stop:
    try:
        # --- 1. Read Sensors ---
        roll, pitch, yaw = imu.getRollPitchYaw(); altitude = gps.getValues()[2]
        roll_rate, pitch_rate, yaw_rate = gyro.getValues()

        # --- 2. Safety & Timeout Checks ---
        if abs(roll) > 0.8 or abs(pitch) > 0.8: emergency_stop("Extreme angles"); continue
        if altitude < -0.1 : emergency_stop("Below ground"); continue
        if not state.emergency_stop and state.should_abort_phase():
            if not state.emergency_stop: print(f"Timeout -> SCAN"); state.phase = "scan"; state.reset_phase_timer(); state.target_lost_frames = 0

        # --- 3. Determine Target Altitude (used by standard PID) ---
        current_target_altitude = TARGET_ALTITUDE
        if state.phase == "hover_low" or (state.phase == "land" and not state.target_found_bottom): # Also use 1m target in land phase if target lost
             current_target_altitude = HOVER_LOW_ALTITUDE

        # --- 4. Common PID Calculations ---
        # Altitude PID based on current target (used if not in landing descent)
        clamped_diff_alt = clamp(current_target_altitude - altitude + K_VERTICAL_OFFSET, -1.0, 1.0)
        vertical_input_pid = K_VERTICAL_P * (clamped_diff_alt ** 3)
        # Base stabilization
        roll_clamped = clamp(roll, -0.5, 0.5); pitch_clamped = clamp(pitch, -0.5, 0.5)
        roll_stabilization = K_ROLL_P * roll_clamped + roll_rate
        pitch_stabilization = K_PITCH_P * pitch_clamped + pitch_rate

        # --- 5. Default Input Values ---
        vertical_thrust = K_VERTICAL_THRUST + vertical_input_pid # Default for non-landing-descent
        roll_input = roll_stabilization; pitch_input = pitch_stabilization; yaw_input = 0.0

        # --- 6. FSM: Camera Selection ---
        current_camera = front_camera; state.active_camera_name = front_camera.getName() # Default front
        if state.phase == "land" and bottom_camera: # Land phase ALWAYS tries to use bottom cam
            current_camera = bottom_camera; state.active_camera_name = bottom_camera.getName()

        # --- 7. FSM: Image Processing ---
        bgr_img, debug_img, found, cx, cy, area, bbox = process_camera_image(current_camera, state.active_camera_name)
        cam_w, cam_h = (current_camera.getWidth(), current_camera.getHeight()) if current_camera else (0, 0)
        # Update target found status (check camera name)
        if bgr_img is not None and current_camera:
             is_front = current_camera.getName() == front_camera.getName()
             is_bottom = bottom_camera and current_camera.getName() == bottom_camera.getName()
             if is_front: state.target_found_front = found
             if is_bottom: state.target_found_bottom = found
        else: found = False; state.target_found_front = False; state.target_found_bottom = False; cx,cy,area,bbox=0,0,0,None

        # --- 8. FSM: Phase Logic ---

        if state.phase == "takeoff": # Target TARGET_ALTITUDE
             if altitude >= TARGET_ALTITUDE - ALTITUDE_TOLERANCE: print("-> SCAN"); state.phase = "scan"; state.reset_phase_timer(); state.target_lost_frames = 0
             # Uses default thrust & stabilization from step 5

        elif state.phase == "scan": # Target TARGET_ALTITUDE, add yaw
             if state.target_found_front:
                  state.target_lost_frames = 0; offset_x = cx - (cam_w // 2)
                  if abs(offset_x) < CENTERING_THRESHOLD: print("-> APPROACH"); state.phase = "approach"; state.reset_phase_timer(); state.target_lost_frames = 0
                  else: yaw_input = clamp(-SCAN_YAW_GAIN * (offset_x / (cam_w // 2)), -0.5, 0.5)
             else:
                  state.target_lost_frames += 1
                  if state.target_lost_frames > 5: yaw_input = SCAN_LOST_TARGET_YAW
             # Uses default thrust & stabilization

        elif state.phase == "approach": # Target TARGET_ALTITUDE, move forward
             if state.target_found_front:
                  state.target_lost_frames = 0; offset_x = cx - (cam_w // 2)
                  relative_area = area / (cam_w * cam_h) if (cam_w * cam_h) > 0 else 0
                  is_centered = abs(offset_x) < OVERHEAD_CENTERING_THRESHOLD
                  is_close = relative_area > OVERHEAD_PROXIMITY_AREA
                  # Check 1: Switch to HOVER_LOW
                  if is_centered and is_close: print("-> HOVER_LOW"); state.phase = "hover_low"; state.reset_phase_timer(); state.target_lost_frames = 0; continue
                  # Check 2: Emergency Stop
                  elif relative_area > EMERGENCY_STOP_AREA_THRESHOLD_FRONT: print("! STOP (Area)"); emergency_stop(f"Front area {relative_area:.2%}"); continue
                  # Check 3: Continue Approach
                  else:
                       speed_scale = relative_area / OVERHEAD_PROXIMITY_AREA if OVERHEAD_PROXIMITY_AREA > 0 else 1.0
                       area_factor = 1.0 - min(speed_scale, APPROACH_AREA_SPEED_FACTOR)
                       forward_tilt = APPROACH_TILT * area_factor
                       pitch_input = pitch_stabilization + K_PITCH_P * clamp(forward_tilt, -0.3, 0.0)
                       yaw_input = clamp(-APPROACH_YAW_GAIN * (offset_x / (cam_w // 2)), -0.3, 0.3)
             else: # Target lost
                  state.target_lost_frames += 1
                  if state.target_lost_frames > APPROACH_LOST_TARGET_FRAMES: print("! Lost -> SCAN"); state.phase = "scan"; state.reset_phase_timer()
             # Uses default thrust & roll

        elif state.phase == "hover_low": # Target HOVER_LOW_ALTITUDE, use front cam
             # Uses default thrust (PID to 1m) and stabilization
             print(f"[HOVER_LOW] Alt:{altitude:.2f}/{HOVER_LOW_ALTITUDE:.1f} ", end="")
             if state.target_found_front:
                  state.target_lost_frames = 0; offset_x = cx - (cam_w // 2); offset_y = cy - (cam_h // 2)
                  # Fine centering
                  yaw_input = clamp(-APPROACH_YAW_GAIN * offset_x/(cam_w//2), -0.2, 0.2) # Fine yaw
                  pitch_correction = clamp(-HOVER_LOW_CENTERING_P * offset_y, -HOVER_LOW_MAX_CORRECTION, HOVER_LOW_MAX_CORRECTION)
                  pitch_input += K_PITCH_P * pitch_correction # Add correction
                  print(f"Center(X:{offset_x},Y:{offset_y}) ", end="")
                  # Check transition to LAND
                  alt_ok = abs(altitude - HOVER_LOW_ALTITUDE) < ALTITUDE_TOLERANCE
                  center_ok = abs(offset_x) < OVERHEAD_CENTERING_THRESHOLD # Check centering
                  if alt_ok and center_ok:
                       print("-> LAND")
                       if bottom_camera: state.phase = "land"; state.reset_phase_timer(); state.target_lost_frames = 0; continue
                       else: print("! No Btm Cam"); emergency_stop("No bottom cam"); continue
                  else: print(f"Stblz(AltOK:{alt_ok},CtrOK:{center_ok})")
             else: # Target lost
                  state.target_lost_frames += 1; print(f"Lost({state.target_lost_frames})", end="")
                  if state.target_lost_frames > APPROACH_LOST_TARGET_FRAMES: print("! Lost -> SCAN"); state.phase = "scan"; state.reset_phase_timer()
                  print("") # Newline after status print

        # ===========================================================
        # MODIFIED LAND PHASE LOGIC
        # ===========================================================
        elif state.phase == "land":
             # Always uses bottom camera (selected in step 6)
             if not bottom_camera: print("! LAND No Btm Cam -> SCAN"); state.phase = "scan"; state.reset_phase_timer(); continue

             log_msg = f"[LAND] Alt:{altitude:.2f} "
             # --- Check if bottom camera sees target ---
             if state.target_found_bottom:
                  # --- TARGET FOUND: Perform Landing Descent & Centering ---
                  state.target_lost_frames = 0
                  # Calculate landing thrust (decays with altitude)
                  vertical_thrust = clamp(LANDING_THRUST_BASE + LANDING_VERTICAL_P * altitude, 0, K_VERTICAL_THRUST * 1.1)
                  # Calculate centering corrections based on bottom camera
                  offset_x = cx - (cam_w // 2); offset_y = cy - (cam_h // 2)
                  roll_correction = clamp(LANDING_CENTERING_P * offset_x, -LANDING_MAX_CORRECTION_TILT, LANDING_MAX_CORRECTION_TILT)
                  pitch_correction = clamp(-LANDING_CENTERING_P * offset_y, -LANDING_MAX_CORRECTION_TILT, LANDING_MAX_CORRECTION_TILT)
                  # Add corrections to stabilization inputs
                  roll_input = roll_stabilization + K_ROLL_P * roll_correction
                  pitch_input = pitch_stabilization + K_PITCH_P * pitch_correction
                  yaw_input = 0 # Keep heading stable
                  log_msg += f"Target Found: Descending... Offs({offset_x},{offset_y})"

             else:
                  # --- TARGET NOT FOUND: Hold Altitude at ~1m ---
                  state.target_lost_frames += 1
                  # Use standard PID to hold HOVER_LOW_ALTITUDE (uses vertical_input_pid calculated in step 4 based on current_target_altitude)
                  vertical_thrust = K_VERTICAL_THRUST + vertical_input_pid
                  # Use stabilization only, no centering correction
                  roll_input = roll_stabilization
                  pitch_input = pitch_stabilization
                  yaw_input = 0
                  log_msg += f"Target Lost({state.target_lost_frames}): Holding Alt @ {HOVER_LOW_ALTITUDE:.1f}m"
                  # Optional: Revert to scan if target lost for too long while trying to land
                  if state.target_lost_frames > LAND_LOST_TARGET_FRAMES_TIMEOUT:
                       print("! Lost Target too long in LAND -> SCAN")
                       state.phase = "scan"; state.reset_phase_timer()
                       continue

             print(log_msg)
             # Check if physically landed (only possible if descending)
             if altitude < LANDED_ALTITUDE_THRESHOLD:
                  land_complete(); continue
        # ===========================================================
        # END OF MODIFIED LAND PHASE
        # ===========================================================

        # --- 9. Final Motor Commands ---
        if not state.emergency_stop:
            FL = vertical_thrust - roll_input + pitch_input - yaw_input; FR = vertical_thrust + roll_input + pitch_input + yaw_input
            RL = vertical_thrust - roll_input - pitch_input + yaw_input; RR = vertical_thrust + roll_input - pitch_input - yaw_input
            motors[0].setVelocity(FL); motors[1].setVelocity(-FR); motors[2].setVelocity(-RL); motors[3].setVelocity(RR)

        # --- 10. OpenCV Window Update ---
        if cv2.waitKey(1) & 0xFF == ord('q'): print("User exit."); break

    # --- Error Handling ---
    except Exception as e: print(f"\n❌ MAIN LOOP ERROR: {e} ❌"); import traceback; traceback.print_exc(); emergency_stop("Loop Exception"); break

# --- Cleanup ---
print("Simulation ended.");
if not state.emergency_stop: emergency_stop("End script")
cv2.destroyAllWindows()