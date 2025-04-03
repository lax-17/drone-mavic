from controller import Robot, GPS, InertialUnit, Gyro, Motor, Receiver
import cv2
import numpy as np
import math
import time



TIME_STEP = 32

# Safety parameters
MAX_SCAN_TIME = 30  # seconds
MAX_APPROACH_TIME = 20  # seconds
MIN_SAFE_DISTANCE = 0.5  # meters
EMERGENCY_STOP_DISTANCE = 0.3  # meters
TARGET_UPDATE_INTERVAL = 0.5  # seconds between GPS updates

# Control gains
K_VERTICAL_THRUST = 68.0      # Slightly reduced from 70.0
K_VERTICAL_OFFSET = 0.6
K_VERTICAL_P      = 3.0
K_ROLL_P          = 15.0      # Reduced from 20.0 for more stability      
K_PITCH_P         = 15.0      # Reduced from 20.0 for more stability

# Position control gains
K_POSITION_P = 0.2  # NEW Reduced Position error gain
K_YAW_P = 0.8      # Yaw error gain
K_ALTITUDE_P = 0.5 # Altitude error gain

# Altitude
TARGET_ALTITUDE = 3.0
LANDING_APPROACH_ALTITUDE = 1.0 # Start landing when below this altitude in approach phase
LANDING_TARGET_ALTITUDE = 1.5     # NEW Target altitude based on observed car GPS Z + buffer
LANDING_DESCENT_SPEED = 0.5     # NEW Reduced descent speed during landing
LANDING_HORIZONTAL_P = 0.5      # NEW Reduced proportional gain for horizontal adjustments during landing


SCAN_YAW_GAIN      = 0.3      # Reduced from 1.0 for stability
CENTERING_THRESHOLD = 50      # Increased from 40 for smoother transitions
APPROACH_YAW_GAIN  = 0.05     # Reduced from 0.1 for smoother yaw control
APPROACH_TILT      = -0.15    # Reduced from -0.25 for gentler approach

# Minimum contour area to consider a valid target
MIN_TARGET_SIZE = 50

#Color detection: wide red ranges
RED_HSV_RANGES = [
    ((0,   50,  50), (10,  255, 255)),
    ((170, 50,  50), (180, 255, 255))
]

# State tracking
class DroneState:
    def __init__(self):
        self.phase = "takeoff"
        self.target_found = False
        self.last_target_time = 0
        self.target_lost_frames = 0
        self.phase_start_time = time.time()
        self.emergency_stop = False
        self.last_gps_update = 0
        self.target_position = None  # (x, y, z) in world coordinates
        self.use_vision_backup = False
        self.landing_initiated = False # Flag to ensure landing transition happens once

    def reset_phase_timer(self):
        self.phase_start_time = time.time()
        self.landing_initiated = False # Reset landing flag when phase changes

    def get_phase_duration(self):
        return time.time() - self.phase_start_time

    def should_abort_phase(self):
        if self.phase == "scan" and self.get_phase_duration() > MAX_SCAN_TIME:
            return True
        if self.phase == "approach" and self.get_phase_duration() > MAX_APPROACH_TIME:
            return True
        return False

    def update_target_position(self, position):
        current_time = time.time()
        if current_time - self.last_gps_update >= TARGET_UPDATE_INTERVAL:
            self.target_position = position
            self.last_gps_update = current_time
            self.target_found = True
            self.target_lost_frames = 0

# Initialize robot and sensors
try:
    robot = Robot()
    gps = robot.getDevice("drone_gps")
    imu = robot.getDevice("drone_imu")
    gyro = robot.getDevice("gyro")
    camera = robot.getDevice("camera")
    receiver = robot.getDevice("receiver")  # Add receiver for car GPS

    gps.enable(TIME_STEP)
    imu.enable(TIME_STEP)
    gyro.enable(TIME_STEP)
    camera.enable(TIME_STEP)
    receiver.enable(TIME_STEP)  # Enable receiver

    motor_names = ["front left propeller", "front right propeller",
                   "rear left propeller", "rear right propeller"]
    motors = [robot.getDevice(name) for name in motor_names]
    for m in motors:
        m.setPosition(float("inf"))
        m.setVelocity(0.0)

    state = DroneState()
except Exception as e:
    print(f"Error initializing robot: {e}")
    exit(1)

#OPENCV WINDOWS
cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
# cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
# cv2.namedWindow("HSV", cv2.WINDOW_NORMAL)


def clamp(value, min_val, max_val):
    return max(min_val, min(value, max_val))

def stop_motors():
    """Stop all motors immediately"""
    for m in motors:
        m.setVelocity(0.0)
    print("⚠️ MOTORS STOPPED")
    state.emergency_stop = True

def detect_red_car(image_bgr):
    """
    Returns (found, cx, cy, area, bbox)
      found: bool
      cx, cy: center of bounding box
      area: contour area
      bbox: (x1, y1, x2, y2)
    """
    try:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        # cv2.imshow("HSV", hsv)

        # Combined mask for all red ranges
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for (lower, upper) in RED_HSV_RANGES:
            lower_np = np.array(lower)
            upper_np = np.array(upper)
            current_mask = cv2.inRange(hsv, lower_np, upper_np)
            mask |= current_mask

       
        # kernel = np.ones((3,3), np.uint8)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        # cv2.imshow("Mask", mask)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        debug_img = image_bgr.copy()
        cv2.drawContours(debug_img, contours, -1, (0, 255, 0), 1)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            if area > MIN_TARGET_SIZE:
                x, y, w, h = cv2.boundingRect(largest)
                cx = x + w//2
                cy = y + h//2
                cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.circle(debug_img, (cx, cy), 5, (255, 0, 0), -1)
                cv2.putText(debug_img, f"Area: {area:.0f}", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                cv2.imshow("Original", debug_img)
                return True, cx, cy, area, (x, y, x+w, y+h)

        cv2.imshow("Original", debug_img)
        return False, 0, 0, 0, None
    except Exception as e:
        print(f"Error in detect_red_car: {e}")
        return False, 0, 0, 0, None

def calculate_position_error(current_pos, target_pos):
    """Calculate distance, bearing to target, and altitude difference."""
    dx = target_pos[0] - current_pos[0]
    dy = target_pos[1] - current_pos[1]
    dz = target_pos[2] - current_pos[2]
    
    distance = math.sqrt(dx*dx + dy*dy)
    # target_bearing is the angle in the drone's local XY plane to the target
    target_bearing = math.atan2(dy, dx)
    
    return distance, target_bearing, dz

def gps_to_local_coordinates(gps_values, target_gps):
    """Convert GPS coordinates to local coordinates"""
    # Simple conversion - can be improved with proper coordinate transformation
    dx = target_gps[0] - gps_values[0]
    dy = target_gps[1] - gps_values[1]
    dz = target_gps[2] - gps_values[2]
    return (dx, dy, dz)

def update_car_gps(car_gps_values):
    """Update the target position with car's GPS coordinates"""
    if car_gps_values is not None and len(car_gps_values) >= 3:
        print(f"✅ [update_car_gps] Valid data received: {car_gps_values}")
        state.update_target_position(car_gps_values)
    else:
        print(f"⚠️ [update_car_gps] Invalid car GPS data received: {car_gps_values}")

while robot.step(TIME_STEP) != -1:
    try:
        # Read sensors
        roll, pitch, yaw = imu.getRollPitchYaw()
        gps_values = gps.getValues()
        altitude = gps_values[2]
        roll_rate, pitch_rate, yaw_rate = gyro.getValues()

        # Check for car GPS data
        queue_length = receiver.getQueueLength()
        if queue_length > 0:
            # car_gps_data = receiver.getData() # Incorrect: Tries to decode as string
            car_gps_data = receiver.getBytes() # Correct: Get raw bytes
            print(f"📬 Received packet: type={type(car_gps_data)}, len={len(car_gps_data)}") # DEBUG type and length
            # Ensure data length is correct for 3 float64s (3 * 8 bytes = 24 bytes)
            if len(car_gps_data) == 24:
                 car_gps = np.frombuffer(car_gps_data, dtype=np.float64) # Parse bytes into numpy array
                 print(f"✅ Parsed GPS: {car_gps}") # DEBUG parsed array
                 state.update_target_position(car_gps) # Update state with numpy array
            else:
                 print(f"❌ Incorrect data length received: {len(car_gps_data)} bytes (expected 24)")
            receiver.nextPacket()
        # else:
            # print("📪 [Main Loop] Receiver queue empty.")

        # Safety check for extreme angles
        if abs(roll) > 0.8 or abs(pitch) > 0.8:
            print("⚠️ Extreme angles detected - stopping motors")
            stop_motors()
            break

        # Common altitude PID
        clamped_diff_alt = clamp(TARGET_ALTITUDE - altitude + K_VERTICAL_OFFSET, -1.0, 1.0)
        vertical_input = K_VERTICAL_P * (clamped_diff_alt ** 3)

        # Clamp roll & pitch angles
        roll_clamped  = clamp(roll,  -0.5,  0.5)
        pitch_clamped = clamp(pitch, -0.5,  0.5)

        # Base PID for roll & pitch
        roll_input  = K_ROLL_P  * roll_clamped  + roll_rate
        pitch_input = K_PITCH_P * pitch_clamped + pitch_rate # Base pitch for stabilization

        # Check for phase timeout
        if state.should_abort_phase():
            print(f"⚠️ Phase {state.phase} timeout - returning to scan")
            state.phase = "scan"
            state.reset_phase_timer()

        if state.phase == "takeoff":
            if altitude < TARGET_ALTITUDE - 0.1:
                front_left  = K_VERTICAL_THRUST + vertical_input - roll_input + pitch_input
                front_right = K_VERTICAL_THRUST + vertical_input + roll_input + pitch_input
                rear_left   = K_VERTICAL_THRUST + vertical_input - roll_input - pitch_input
                rear_right  = K_VERTICAL_THRUST + vertical_input + roll_input - pitch_input

                motors[0].setVelocity(front_left)
                motors[1].setVelocity(-front_right)
                motors[2].setVelocity(-rear_left)
                motors[3].setVelocity(rear_right)

                print(f"[TAKEOFF] Altitude: {altitude:.2f}")
            else:
                print("🟢 Altitude reached. Switching to SCAN phase.")
                state.phase = "scan"
                state.reset_phase_timer()

        elif state.phase == "scan":
            print(f"ℹ️ [Scan Phase] Checking GPS. Target position: {state.target_position}")
            # Initialize yaw_input for this timestep
            yaw_input = 0 
            # Reset pitch_input to base stabilization for this timestep
            pitch_input = K_PITCH_P * pitch_clamped + pitch_rate
            
            if state.target_position is not None:
                # --- GPS Scan Logic ---
                drone_current_yaw = yaw # Get current yaw from IMU read earlier
                local_pos = gps_to_local_coordinates(gps_values, state.target_position)
                # Get distance and bearing (angle) to target relative to drone's current position/orientation
                distance, target_bearing, alt_error = calculate_position_error((0,0,0), local_pos)
                
                if distance < MIN_SAFE_DISTANCE:
                    print("🎯 Target within safe distance → Approach phase")
                    state.phase = "approach"
                    state.reset_phase_timer()
                else:
                    # Calculate heading error: difference between target bearing and current drone heading
                    heading_error = target_bearing - drone_current_yaw
                    # Normalize the error to the range [-pi, pi] to ensure shortest turn
                    while heading_error > math.pi: heading_error -= 2 * math.pi
                    while heading_error < -math.pi: heading_error += 2 * math.pi
                        
                    # Calculate yaw input based on heading error
                    yaw_input = K_YAW_P * heading_error
                    
                    # --- Add Forward Motion for Scan Phase --- 
                    # Apply forward tilt based on distance, clamped (similar to approach)
                    # Negative tilt moves forward. Clamp distance effect to prevent excessive speed.
                    forward_tilt = -K_POSITION_P * clamp(distance, 0.0, 1.0) # RE-ENABLED: Match approach phase clamping
                    pitch_input = K_PITCH_P * clamp(pitch + forward_tilt, -0.5, 0.5) + pitch_rate # RE-ENABLED: Add tilt to base pitch
                    # --- TEMPORARILY DISABLED FORWARD TILT IN SCAN --- 
                    # forward_tilt = 0 # Explicitly set to 0 for now
                    # pitch_input = K_PITCH_P * pitch_clamped + pitch_rate # Use only base stabilization pitch
                    
                    print(f"🔄 GPS Scanning: dist={distance:.2f}, bearing={target_bearing:.2f}, drone_yaw={drone_current_yaw:.2f}, head_err={heading_error:.2f}, yaw_in={yaw_input:.2f}, tilt={forward_tilt:.2f}") # Updated debug info
                print(f"🛰️ [Scan Phase] Using GPS.")
            else:
                # --- Vision Scan Logic ---
                print(f"👁️ [Scan Phase] No GPS data, using vision.")
                img = camera.getImage()
                if img:
                    w = camera.getWidth()
                    h = camera.getHeight()
                    img_array = np.frombuffer(img, np.uint8).reshape((h, w, 4))
                    bgr_img   = cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)

                    found, cx, cy, area, bbox = detect_red_car(bgr_img)
                    if found:
                        state.target_found = True
                        state.last_target_time = time.time()
                        state.target_lost_frames = 0
                        
                        offset = cx - (w // 2)
                        if abs(offset) < CENTERING_THRESHOLD:
                            state.phase = "approach"
                            state.reset_phase_timer()
                            print("🎯 Target centered → Approach phase")
                        else:
                            yaw_input = -SCAN_YAW_GAIN * (offset / (w // 2)) # Vision yaw input
                            print(f"🔄 Vision Scanning: offset={offset}, yaw_input={yaw_input:.2f}")
                    else:
                        state.target_lost_frames += 1
                        if state.target_lost_frames > 10:
                            state.target_found = False
                            yaw_input = 0.15 # Vision rotation input
                            print("🔍 No target found, rotating...")

            # --- Apply Motor Commands (Scan) ---
            # Motors commands use the calculated inputs (pitch_input now includes forward tilt if GPS active)
            front_left  = K_VERTICAL_THRUST + vertical_input - roll_input + pitch_input - yaw_input
            front_right = K_VERTICAL_THRUST + vertical_input + roll_input + pitch_input + yaw_input
            rear_left   = K_VERTICAL_THRUST + vertical_input - roll_input - pitch_input + yaw_input
            rear_right  = K_VERTICAL_THRUST + vertical_input + roll_input - pitch_input - yaw_input

            motors[0].setVelocity(front_left)
            motors[1].setVelocity(-front_right)
            motors[2].setVelocity(-rear_left)
            motors[3].setVelocity(rear_right)

        elif state.phase == "approach":
            print(f"ℹ️ [Approach Phase] Checking GPS. Target position: {state.target_position}")
            initiate_landing = False # Flag to check if we should land

            if state.target_position is not None:
                # --- GPS Approach Logic ---
                drone_current_yaw = yaw # Get current yaw from IMU
                local_pos = gps_to_local_coordinates(gps_values, state.target_position)
                distance, target_bearing, alt_error = calculate_position_error((0,0,0), local_pos)
                print(f"🛰️ [Approach Phase] Using GPS. Dist: {distance:.2f}, Alt: {altitude:.2f}, bearing: {target_bearing:.2f}, drone_yaw: {drone_current_yaw:.2f}")

                # Calculate heading error for yaw control
                heading_error = target_bearing - drone_current_yaw
                # Normalize the error to the range [-pi, pi]
                while heading_error > math.pi:
                    heading_error -= 2 * math.pi
                while heading_error < -math.pi:
                    heading_error += 2 * math.pi
                    
                # --- Horizontal Control ---
                yaw_input = K_YAW_P * heading_error # Use heading_error for yaw
                # Apply forward tilt based on distance, clamped
                forward_tilt = -K_POSITION_P * clamp(distance, 0.0, 1.0) # Negative tilt moves forward
                pitch_input = K_PITCH_P * clamp(pitch + forward_tilt, -0.5, 0.5) + pitch_rate
                # Roll input primarily for stabilization unless there's lateral error (not calculated here)
                # roll_input stays as base stabilization

                # --- Landing Check (GPS) ---
                # Check if close enough horizontally and low enough vertically
                if distance < MIN_SAFE_DISTANCE and altitude < LANDING_APPROACH_ALTITUDE:
                    print(f"✅ [Approach Phase] GPS landing conditions met. Dist: {distance:.2f}, Alt: {altitude:.2f}")
                    initiate_landing = True

                # --- Safety Stop (GPS) ---
                if distance < EMERGENCY_STOP_DISTANCE:
                    print("⚠️ Target too close (GPS) - stopping motors")
                    stop_motors()
                    break

            else:
                # --- Vision Approach Logic ---
                print(f"👁️ [Approach Phase] No GPS data, using vision.")
                img = camera.getImage()
                if img:
                    w = camera.getWidth()
                    h = camera.getHeight()
                    img_array = np.frombuffer(img, np.uint8).reshape((h, w, 4))
                    bgr_img   = cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)

                    found, cx, cy, area, bbox = detect_red_car(bgr_img)
                    if found:
                        state.target_found = True
                        state.last_target_time = time.time()
                        state.target_lost_frames = 0

                        # --- Horizontal Control (Vision) ---
                        offset_x = cx - (w // 2)
                        offset_y = cy - (h // 2) # Add vertical offset check if needed
                        area_norm = area / (w * h) # Normalized area 0 to 1

                        # Yaw based on horizontal offset
                        yaw_input = -APPROACH_YAW_GAIN * (offset_x / (w // 2))
                        # Pitch (forward/backward) based on area/estimated distance
                        # Less tilt as area gets larger (closer)
                        area_factor = clamp(1.0 - (area_norm / 0.20), 0.0, 1.0) # Approach stops if area > 20%
                        forward_tilt = APPROACH_TILT * area_factor # Negative tilt moves forward
                        pitch_input = K_PITCH_P * clamp(pitch + forward_tilt, -0.5, 0.5) + pitch_rate
                        # Roll input could be added based on offset_x if needed, but yaw handles centering
                        print(f"[APPROACH] Vision offset={offset_x}, area%={area_norm*100:.1f}, yaw={yaw_input:.2f}, tilt={forward_tilt:.2f}")

                        # --- Landing Check (Vision) ---
                        # Land if centered, low enough, and reasonably large area
                        if abs(offset_x) < CENTERING_THRESHOLD and altitude < LANDING_APPROACH_ALTITUDE and area_norm > 0.05: # e.g. area > 5%
                            print(f"✅ [Approach Phase] Vision landing conditions met. Offset: {offset_x}, Alt: {altitude:.2f}, Area: {area_norm*100:.1f}%")
                            initiate_landing = True

                        # --- Safety Stop (Vision) ---
                        if area_norm > 0.4: # Stop if target takes > 40% of view
                            print("⚠️ Target too close (vision) - stopping motors")
                            stop_motors()
                            break
                    else:
                        state.target_lost_frames += 1
                        print(f"❓ [Approach Phase] Vision Target lost ({state.target_lost_frames} frames)")
                        if state.target_lost_frames > 20: # Increased tolerance
                            print("⚠️ Target lost in approach → returning to SCAN")
                            state.phase = "scan"
                            state.reset_phase_timer()
                            # Reset inputs if target lost
                            yaw_input = 0
                            pitch_input = K_PITCH_P * pitch_clamped + pitch_rate
                            roll_input  = K_ROLL_P  * roll_clamped  + roll_rate

            # --- Initiate Landing Transition ---
            if initiate_landing and not state.landing_initiated:
                print("🛬 Transitioning to LAND phase.")
                state.phase = "land"
                state.landing_initiated = True # Prevent multiple transitions
                state.reset_phase_timer() # Reset timer for landing phase
                # Use current inputs for the first step of landing
            else:
                 # --- Apply Motor Commands (Approach) ---
                # Only apply if not transitioning to land this step
                front_left  = K_VERTICAL_THRUST + vertical_input - roll_input + pitch_input - yaw_input
                front_right = K_VERTICAL_THRUST + vertical_input + roll_input + pitch_input + yaw_input
                rear_left   = K_VERTICAL_THRUST + vertical_input - roll_input - pitch_input + yaw_input
                rear_right  = K_VERTICAL_THRUST + vertical_input + roll_input - pitch_input - yaw_input

                motors[0].setVelocity(front_left)
                motors[1].setVelocity(-front_right)
                motors[2].setVelocity(-rear_left)
                motors[3].setVelocity(rear_right)

        elif state.phase == "land":
             print(f"🛬 [Land Phase] Altitude: {altitude:.2f}, Target Alt: {LANDING_TARGET_ALTITUDE}")

             # --- Check for Landing Completion ---
             if altitude <= LANDING_TARGET_ALTITUDE:
                 print("✅ Landed! Stopping motors.")
                 stop_motors()
                 break # Exit simulation loop

             # --- Landing Control Calculations ---
             # Vertical Control: Descend at LANDING_DESCENT_SPEED
             # Simple approach: Reduce base thrust slightly, remove upward PID correction
             # More robust: PID control to target descent speed based on gyro/altitude change.
             # Let's try a simpler approach first:
             vertical_input = K_VERTICAL_P * clamp(LANDING_TARGET_ALTITUDE - altitude, -1.0, 0.0) # Only allow negative correction (downwards)
             # Reduce base thrust slightly to encourage descent
             landing_thrust = K_VERTICAL_THRUST * LANDING_DESCENT_SPEED

             # Horizontal Control: Maintain position over target
             yaw_input = 0 # Keep yaw steady during final descent (or use last known yaw_error)
             pitch_input = K_PITCH_P * pitch_clamped + pitch_rate # Base stabilization
             roll_input  = K_ROLL_P  * roll_clamped  + roll_rate # Base stabilization

             if state.target_position is not None:
                 # Fine-tune position using GPS if available
                 # local_pos = gps_to_local_coordinates(gps_values, state.target_position)
                 # distance, yaw_error, alt_error = calculate_position_error((0,0,0), local_pos) # OLD
                 # print(f"🛰️ [Land Phase] Using GPS. Dist: {distance:.2f}, YawErr: {yaw_error:.2f}") # OLD

                 # Use small P gains for gentle correction during landing
                 # yaw_input = LANDING_HORIZONTAL_P * clamp(yaw_error, -0.5, 0.5) # OLD/INCORRECT
                 # forward_tilt = -LANDING_HORIZONTAL_P * clamp(distance, 0.0, 0.5) # Gentle forward tilt
                 # pitch_input = K_PITCH_P * clamp(pitch + forward_tilt, -0.5, 0.5) + pitch_rate
                 
                 # --- Corrected GPS Landing Logic ---
                 drone_current_yaw = yaw # Get current yaw
                 local_pos = gps_to_local_coordinates(gps_values, state.target_position)
                 distance, target_bearing, alt_error = calculate_position_error((0,0,0), local_pos)

                 # Calculate heading error for yaw control
                 heading_error = target_bearing - drone_current_yaw
                 # Normalize the error to the range [-pi, pi]
                 while heading_error > math.pi: heading_error -= 2 * math.pi
                 while heading_error < -math.pi: heading_error += 2 * math.pi

                 print(f"🛰️ [Land Phase] Using GPS. Dist: {distance:.2f}, HeadErr: {heading_error:.2f}")

                 # Use small P gains for gentle correction during landing
                 yaw_input = LANDING_HORIZONTAL_P * heading_error # CORRECT: Use heading_error
                 # Clamp max yaw input during landing for stability
                 yaw_input = clamp(yaw_input, -0.3, 0.3)
                 forward_tilt = -LANDING_HORIZONTAL_P * clamp(distance, 0.0, 0.5) # Gentle forward tilt
                 pitch_input = K_PITCH_P * clamp(pitch + forward_tilt, -0.5, 0.5) + pitch_rate
                 # Roll correction could be added if dx/dy are separated in calculate_position_error

             else:
                 # Vision fallback during landing (use last known good center or keep camera active)
                 # For simplicity, we'll just use base stabilization if GPS is lost during landing phase.
                 print(f"👁️ [Land Phase] No GPS, using stabilization only.")
                 # Keep yaw_input = 0, pitch/roll use base stabilization

             # --- Apply Motor Commands (Land) ---
             front_left  = landing_thrust + vertical_input - roll_input + pitch_input - yaw_input
             front_right = landing_thrust + vertical_input + roll_input + pitch_input + yaw_input
             rear_left   = landing_thrust + vertical_input - roll_input - pitch_input + yaw_input
             rear_right  = landing_thrust + vertical_input + roll_input - pitch_input - yaw_input

             motors[0].setVelocity(front_left)
             motors[1].setVelocity(-front_right)
             motors[2].setVelocity(-rear_left)
             motors[3].setVelocity(rear_right)

        # Let OpenCV update windows; press 'q' to quit simulation
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e: # Outer exception handler
        print(f"❌ Error in main loop: {e}") # Print the error caught here
        import traceback # Import traceback module
        traceback.print_exc() # Print the full traceback
        stop_motors()
        break

cv2.destroyAllWindows()