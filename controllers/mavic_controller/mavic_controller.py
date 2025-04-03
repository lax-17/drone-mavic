from controller import Robot, GPS, InertialUnit, Gyro, Motor
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

# Control gains
K_VERTICAL_THRUST = 68.0      # Slightly reduced from 70.0
K_VERTICAL_OFFSET = 0.6
K_VERTICAL_P      = 3.0
K_ROLL_P          = 15.0      # Reduced from 20.0 for more stability      
K_PITCH_P         = 15.0      # Reduced from 20.0 for more stability

# Altitude
TARGET_ALTITUDE = 3.0


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

    def reset_phase_timer(self):
        self.phase_start_time = time.time()

    def get_phase_duration(self):
        return time.time() - self.phase_start_time

    def should_abort_phase(self):
        if self.phase == "scan" and self.get_phase_duration() > MAX_SCAN_TIME:
            return True
        if self.phase == "approach" and self.get_phase_duration() > MAX_APPROACH_TIME:
            return True
        return False

robot = Robot()

gps = robot.getDevice("drone_gps")
imu = robot.getDevice("drone_imu")
gyro = robot.getDevice("gyro")
camera = robot.getDevice("camera")

gps.enable(TIME_STEP)
imu.enable(TIME_STEP)
gyro.enable(TIME_STEP)
camera.enable(TIME_STEP)

motor_names = ["front left propeller", "front right propeller",
               "rear left propeller", "rear right propeller"]
motors = [robot.getDevice(name) for name in motor_names]
for m in motors:
    m.setPosition(float("inf"))
    m.setVelocity(0.0)


state = DroneState()

#OPENCV WINDOWS
cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
# cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
# cv2.namedWindow("HSV", cv2.WINDOW_NORMAL)


def clamp(value, min_val, max_val):
    return max(min_val, min(value, max_val))

def emergency_stop():
    """Stop all motors immediately"""
    for m in motors:
        m.setVelocity(0.0)
    print("⚠️ EMERGENCY STOP ACTIVATED")
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


while robot.step(TIME_STEP) != -1:
    try:
        # Read sensors
        roll, pitch, yaw = imu.getRollPitchYaw()
        altitude = gps.getValues()[2]
        roll_rate, pitch_rate, yaw_rate = gyro.getValues()

        # Safety check for extreme angles
        if abs(roll) > 0.8 or abs(pitch) > 0.8:
            print("⚠️ Extreme angles detected - emergency stop")
            emergency_stop()
            break

        # Common altitude PID (used in all phases)
        clamped_diff_alt = clamp(TARGET_ALTITUDE - altitude + K_VERTICAL_OFFSET, -1.0, 1.0)
        vertical_input = K_VERTICAL_P * (clamped_diff_alt ** 3)

        # We'll clamp the roll & pitch angles to avoid flipping
        roll_clamped  = clamp(roll,  -0.5,  0.5)   # limit max tilt
        pitch_clamped = clamp(pitch, -0.5,  0.5)   # limit max tilt

        # Base PID for roll & pitch
        roll_input  = K_ROLL_P  * roll_clamped  + roll_rate
        pitch_input = K_PITCH_P * pitch_clamped + pitch_rate

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
            # Default no yaw
            yaw_input = 0.0

            # Get camera image
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
                        # Target is roughly centered
                        state.phase = "approach"
                        state.reset_phase_timer()
                        print("🎯 Target centered → Approach phase")
                    else:
                        # Proportional yaw to bring target to center (reversed sign)
                        yaw_input = -SCAN_YAW_GAIN * (offset / (w // 2))  # Added negative sign
                        print(f"🔄 Scanning: offset={offset}, yaw_input={yaw_input:.2f}")
                else:
                    state.target_lost_frames += 1
                    if state.target_lost_frames > 10:  # Lost target for 10 frames
                        state.target_found = False
                        yaw_input = 0.15  # Reduced from 0.3 for more stable rotation
                        print("🔍 Scanning: No target found, rotating...")

            # Compute final motor commands
            front_left  = K_VERTICAL_THRUST + vertical_input - roll_input + pitch_input - yaw_input
            front_right = K_VERTICAL_THRUST + vertical_input + roll_input + pitch_input + yaw_input
            rear_left   = K_VERTICAL_THRUST + vertical_input - roll_input - pitch_input + yaw_input
            rear_right  = K_VERTICAL_THRUST + vertical_input + roll_input - pitch_input - yaw_input

            motors[0].setVelocity(front_left)
            motors[1].setVelocity(-front_right)
            motors[2].setVelocity(-rear_left)
            motors[3].setVelocity(rear_right)


        elif state.phase == "approach":
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
                    # Calculate forward speed based on distance (inversely related to area)
                    area_factor = 1.0 - min(area / (w * h * 0.25), 0.8)  # Limit to 80% of max speed
                    forward_tilt = APPROACH_TILT * area_factor
                    
                    # Add yaw control to keep target centered (reversed sign)
                    yaw_input = -APPROACH_YAW_GAIN * (offset / (w // 2))  # Added negative sign
                    
                    print(f"Speed factor: {area_factor:.2f}, tilt: {forward_tilt:.3f}, yaw: {yaw_input:.3f}")

                    # Safety distance check
                    if area > w * h * 0.4:  # Target too close
                        print("⚠️ Target too close - emergency stop")
                        emergency_stop()
                        break

                    # Recompute pitch_input with forward tilt
                    pitch_input = K_PITCH_P * clamp(pitch + forward_tilt, -0.5, 0.5) + pitch_rate

                    # Include yaw control in motor commands
                    front_left  = K_VERTICAL_THRUST + vertical_input - roll_input + pitch_input - yaw_input
                    front_right = K_VERTICAL_THRUST + vertical_input + roll_input + pitch_input + yaw_input
                    rear_left   = K_VERTICAL_THRUST + vertical_input - roll_input - pitch_input + yaw_input
                    rear_right  = K_VERTICAL_THRUST + vertical_input + roll_input - pitch_input - yaw_input

                    motors[0].setVelocity(front_left)
                    motors[1].setVelocity(-front_right)
                    motors[2].setVelocity(-rear_left)
                    motors[3].setVelocity(rear_right)

                    print(f"[APPROACH] offset={offset}, yaw={yaw_input:.2f}, tilt={forward_tilt:.2f}")
                else:
                    state.target_lost_frames += 1
                    if state.target_lost_frames > 10:
                        print("⚠️ Target lost in approach → returning to SCAN")
                        state.phase = "scan"
                        state.reset_phase_timer()

        # Let OpenCV update windows; press 'q' to quit simulation
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"Error in main loop: {e}")
        emergency_stop()
        break

cv2.destroyAllWindows()