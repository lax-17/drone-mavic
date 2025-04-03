from controller import Robot, GPS, InertialUnit, Gyro, Motor, Receiver
import cv2
import numpy as np
import math
import time
import traceback

# Import custom modules
from state import DroneState
from vision_utils import detect_red_car, CENTERING_THRESHOLD, MIN_TARGET_SIZE # Import necessary items
from control_utils import clamp, calculate_position_error, gps_to_local_coordinates

TIME_STEP = 32

# Safety parameters
MAX_SCAN_TIME = 30  # seconds
MAX_APPROACH_TIME = 20  # seconds
MIN_SAFE_DISTANCE = 0.5  # meters
EMERGENCY_STOP_DISTANCE = 0.3  # meters
TARGET_UPDATE_INTERVAL = 0.5  # seconds between GPS updates

# Control gains
K_VERTICAL_THRUST = 70.0      # NEW Set back to original potential hover value
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
LANDING_DESCENT_SPEED = 1.03    # NEW: Slightly above hover thrust for safety margin
LANDING_HORIZONTAL_P = 0.1      # Keep low for smooth horizontal corrections


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

class DroneController:
    def __init__(self):
        self.robot = Robot()
        self.state = DroneState()
        self.timestep = int(self.robot.getBasicTimeStep()) # Use robot's timestep

        # Initialize Sensors
        self.gps = self.robot.getDevice("drone_gps")
        self.imu = self.robot.getDevice("drone_imu")
        self.gyro = self.robot.getDevice("gyro")
        self.camera = self.robot.getDevice("camera")
        self.receiver = self.robot.getDevice("receiver")

        # Check devices
        if self.gps is None: raise Exception("GPS device 'drone_gps' not found")
        if self.imu is None: raise Exception("IMU device 'drone_imu' not found")
        if self.gyro is None: raise Exception("Gyro device 'gyro' not found")
        if self.camera is None: raise Exception("Camera device 'camera' not found")
        if self.receiver is None: raise Exception("Receiver device 'receiver' not found")

        # Enable Sensors
        self.gps.enable(self.timestep)
        self.imu.enable(self.timestep)
        self.gyro.enable(self.timestep)
        self.camera.enable(self.timestep)
        self.receiver.enable(self.timestep)

        # Initialize Motors
        motor_names = ["front left propeller", "front right propeller",
                       "rear left propeller", "rear right propeller"]
        self.motors = [self.robot.getDevice(name) for name in motor_names]
        for motor in self.motors:
            if motor is None:
                raise Exception(f"Motor device not found (one of {motor_names})")
            motor.setPosition(float('inf'))
            motor.setVelocity(0.0)

        # Sensor values placeholders
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.altitude = 0.0
        self.roll_rate = 0.0
        self.pitch_rate = 0.0
        self.yaw_rate = 0.0
        self.gps_values = None
        self.bgr_image = None # To store the latest camera image
        self.camera_width = self.camera.getWidth()
        self.camera_height = self.camera.getHeight()

        print("DroneController initialized.")
        # Optional: Setup OpenCV windows here if needed for debugging
        # cv2.namedWindow("Original", cv2.WINDOW_NORMAL)

    def _read_sensors(self):
        """Reads current values from onboard sensors."""
        self.roll, self.pitch, self.yaw = self.imu.getRollPitchYaw()
        self.gps_values = self.gps.getValues()
        self.altitude = self.gps_values[2]
        self.roll_rate, self.pitch_rate, self.yaw_rate = self.gyro.getValues()

        # Read camera image
        img_data = self.camera.getImage()
        if img_data:
            img_array = np.frombuffer(img_data, np.uint8).reshape((self.camera_height, self.camera_width, 4))
            self.bgr_image = cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)
            # cv2.imshow("Original", self.bgr_image) # Optional debug view
        else:
            self.bgr_image = None

    def _receive_gps(self):
        """Checks receiver queue and updates target GPS state if data received."""
        queue_length = self.receiver.getQueueLength()
        if queue_length > 0:
            car_gps_data = self.receiver.getBytes() # Use getBytes()
            print(f"📬 Received packet: type={type(car_gps_data)}, len={len(car_gps_data)}")
            if len(car_gps_data) == 24: # 3 * float64 (8 bytes)
                try:
                    car_gps = np.frombuffer(car_gps_data, dtype=np.float64)
                    print(f"✅ Parsed GPS: {car_gps}")
                    self.state.update_target_position(car_gps)
                except Exception as e:
                     print(f"❌ Error parsing GPS buffer: {e}")
            else:
                print(f"❌ Incorrect data length received: {len(car_gps_data)} bytes (expected 24)")
            self.receiver.nextPacket()

    def _apply_motor_velocities(self, front_left, front_right, rear_left, rear_right):
        """Applies calculated velocities to the four motors."""
        self.motors[0].setVelocity(front_left)
        self.motors[1].setVelocity(-front_right) # Note the sign inversion
        self.motors[2].setVelocity(-rear_left)   # Note the sign inversion
        self.motors[3].setVelocity(rear_right)

    def stop_motors(self):
        """Stops all motors immediately."""
        print("⚠️ MOTORS STOPPED")
        for motor in self.motors:
            motor.setVelocity(0.0)
        self.state.emergency_stop = True # Set state flag

    # --- Phase Logic Methods --- 

    def _run_takeoff_phase(self):
        """Handles the takeoff phase until target altitude is reached."""
        print(f"[TAKEOFF] Altitude: {self.altitude:.2f}")
        if self.altitude >= TARGET_ALTITUDE - 0.1:
            print("🟢 Altitude reached. Switching to SCAN phase.")
            self.state.phase = "scan"
            self.state.reset_phase_timer()
            # Apply stabilizing hover thrust on transition frame
            vertical_input = 0 # No vertical correction needed
            roll_input = K_ROLL_P * clamp(self.roll, -0.5, 0.5) + self.roll_rate
            pitch_input = K_PITCH_P * clamp(self.pitch, -0.5, 0.5) + self.pitch_rate
            yaw_input = 0
            self._apply_motor_velocities(
                K_VERTICAL_THRUST + vertical_input - roll_input + pitch_input - yaw_input,
                K_VERTICAL_THRUST + vertical_input + roll_input + pitch_input + yaw_input,
                K_VERTICAL_THRUST + vertical_input - roll_input - pitch_input + yaw_input,
                K_VERTICAL_THRUST + vertical_input + roll_input - pitch_input - yaw_input
            )
        else:
            # Calculate inputs for climbing
            clamped_diff_alt = clamp(TARGET_ALTITUDE - self.altitude + K_VERTICAL_OFFSET, -1.0, 1.0)
            vertical_input = K_VERTICAL_P * (clamped_diff_alt ** 3)
            roll_input = K_ROLL_P * clamp(self.roll, -0.5, 0.5) + self.roll_rate
            pitch_input = K_PITCH_P * clamp(self.pitch, -0.5, 0.5) + self.pitch_rate
            yaw_input = 0 # No yaw during takeoff
            self._apply_motor_velocities(
                K_VERTICAL_THRUST + vertical_input - roll_input + pitch_input - yaw_input,
                K_VERTICAL_THRUST + vertical_input + roll_input + pitch_input + yaw_input,
                K_VERTICAL_THRUST + vertical_input - roll_input - pitch_input + yaw_input,
                K_VERTICAL_THRUST + vertical_input + roll_input - pitch_input - yaw_input
            )

    def _run_scan_phase(self):
        """Handles the scan phase, using GPS first, then vision fallback."""
        print(f"ℹ️ [Scan Phase] Checking GPS. Target position: {self.state.target_position}")
        
        # Base PID inputs for stabilization
        clamped_diff_alt = clamp(TARGET_ALTITUDE - self.altitude + K_VERTICAL_OFFSET, -1.0, 1.0)
        vertical_input = K_VERTICAL_P * (clamped_diff_alt ** 3)
        roll_input = K_ROLL_P * clamp(self.roll, -0.5, 0.5) + self.roll_rate
        pitch_input = K_PITCH_P * clamp(self.pitch, -0.5, 0.5) + self.pitch_rate # Base pitch
        yaw_input = 0 # Default yaw input

        if self.state.target_position is not None:
            # --- GPS Scan Logic ---
            local_pos = gps_to_local_coordinates(self.gps_values, self.state.target_position)
            distance, target_bearing, alt_error = calculate_position_error((0,0,0), local_pos)

            if distance < MIN_SAFE_DISTANCE:
                print("🎯 Target within safe distance → Approach phase")
                self.state.phase = "approach"
                self.state.reset_phase_timer()
                # Use base stabilization inputs for this frame transition
            else:
                # Calculate heading error
                heading_error = target_bearing - self.yaw
                while heading_error > math.pi: heading_error -= 2 * math.pi
                while heading_error < -math.pi: heading_error += 2 * math.pi
                
                yaw_input = K_YAW_P * heading_error
                
                # Add forward motion
                forward_tilt = -K_POSITION_P * clamp(distance, 0.0, 1.0) 
                pitch_input = K_PITCH_P * clamp(self.pitch + forward_tilt, -0.5, 0.5) + self.pitch_rate
                
                print(f"🔄 GPS Scanning: dist={distance:.2f}, bearing={target_bearing:.2f}, drone_yaw={self.yaw:.2f}, head_err={heading_error:.2f}, yaw_in={yaw_input:.2f}, tilt={forward_tilt:.2f}")
            print(f"🛰️ [Scan Phase] Using GPS.")
        else:
            # --- Vision Scan Logic ---
            print(f"👁️ [Scan Phase] No GPS data, using vision.")
            if self.bgr_image is not None:
                found, cx, cy, area, bbox = detect_red_car(self.bgr_image)
                if found:
                    self.state.target_found = True
                    self.state.last_target_time = time.time()
                    self.state.target_lost_frames = 0
                    
                    offset = cx - (self.camera_width // 2)
                    if abs(offset) < CENTERING_THRESHOLD:
                        print("🎯 Target centered (Vision) → Approach phase")
                        self.state.phase = "approach"
                        self.state.reset_phase_timer()
                        # Use base stabilization inputs for this frame transition
                    else:
                        yaw_input = -SCAN_YAW_GAIN * (offset / (self.camera_width // 2))
                        print(f"🔄 Vision Scanning: offset={offset}, yaw_input={yaw_input:.2f}")
                else:
                    self.state.target_lost_frames += 1
                    if self.state.target_lost_frames > 10:
                        self.state.target_found = False
                        yaw_input = 0.15 # Rotate slowly to find target
                        print("🔍 No target found (Vision), rotating...")
            else:
                print("⚠️ No camera image available for vision scan.")

        # Apply motor commands for Scan phase (unless transitioning)
        if self.state.phase == "scan": # Only apply if not changed mid-logic
            self._apply_motor_velocities(
                K_VERTICAL_THRUST + vertical_input - roll_input + pitch_input - yaw_input,
                K_VERTICAL_THRUST + vertical_input + roll_input + pitch_input + yaw_input,
                K_VERTICAL_THRUST + vertical_input - roll_input - pitch_input + yaw_input,
                K_VERTICAL_THRUST + vertical_input + roll_input - pitch_input - yaw_input
            )

    def _run_approach_phase(self):
        """Handles the approach phase, using GPS or vision."""
        print(f"ℹ️ [Approach Phase] Checking GPS. Target position: {self.state.target_position}")

        # Base PID inputs
        clamped_diff_alt = clamp(TARGET_ALTITUDE - self.altitude + K_VERTICAL_OFFSET, -1.0, 1.0)
        vertical_input = K_VERTICAL_P * (clamped_diff_alt ** 3)
        roll_input = K_ROLL_P * clamp(self.roll, -0.5, 0.5) + self.roll_rate
        pitch_input = K_PITCH_P * clamp(self.pitch, -0.5, 0.5) + self.pitch_rate # Base pitch
        yaw_input = 0
        initiate_landing = False

        if self.state.target_position is not None:
            # --- GPS Approach --- 
            local_pos = gps_to_local_coordinates(self.gps_values, self.state.target_position)
            distance, target_bearing, alt_error = calculate_position_error((0,0,0), local_pos)
            print(f"🛰️ [Approach Phase] Using GPS. Dist: {distance:.2f}, Alt: {self.altitude:.2f}, bearing: {target_bearing:.2f}, drone_yaw: {self.yaw:.2f}")

            heading_error = target_bearing - self.yaw
            while heading_error > math.pi: heading_error -= 2 * math.pi
            while heading_error < -math.pi: heading_error += 2 * math.pi
            
            yaw_input = K_YAW_P * heading_error
            forward_tilt = -K_POSITION_P * clamp(distance, 0.0, 1.0)
            pitch_input = K_PITCH_P * clamp(self.pitch + forward_tilt, -0.5, 0.5) + self.pitch_rate

            if distance < MIN_SAFE_DISTANCE and self.altitude < LANDING_APPROACH_ALTITUDE:
                print(f"✅ [Approach Phase] GPS landing conditions met. Dist: {distance:.2f}, Alt: {self.altitude:.2f}")
                initiate_landing = True
            
            if distance < EMERGENCY_STOP_DISTANCE:
                print("⚠️ Target too close (GPS) - stopping motors")
                self.stop_motors()
                return # Exit phase logic early
        else:
             # --- Vision Approach --- 
            print(f"👁️ [Approach Phase] No GPS data, using vision.")
            if self.bgr_image is not None:
                found, cx, cy, area, bbox = detect_red_car(self.bgr_image)
                if found:
                    self.state.target_found = True
                    self.state.last_target_time = time.time()
                    self.state.target_lost_frames = 0

                    offset_x = cx - (self.camera_width // 2)
                    area_norm = area / (self.camera_width * self.camera_height)

                    yaw_input = -APPROACH_YAW_GAIN * (offset_x / (self.camera_width // 2))
                    area_factor = clamp(1.0 - (area_norm / 0.20), 0.0, 1.0) 
                    forward_tilt = APPROACH_TILT * area_factor
                    pitch_input = K_PITCH_P * clamp(self.pitch + forward_tilt, -0.5, 0.5) + self.pitch_rate
                    print(f"[APPROACH] Vision offset={offset_x}, area%={area_norm*100:.1f}, yaw={yaw_input:.2f}, tilt={forward_tilt:.2f}")

                    if abs(offset_x) < CENTERING_THRESHOLD and self.altitude < LANDING_APPROACH_ALTITUDE and area_norm > 0.05: 
                        print(f"✅ [Approach Phase] Vision landing conditions met. Offset: {offset_x}, Alt: {self.altitude:.2f}, Area: {area_norm*100:.1f}%")
                        initiate_landing = True

                    if area_norm > 0.4: 
                        print("⚠️ Target too close (vision) - stopping motors")
                        self.stop_motors()
                        return # Exit phase logic early
                else:
                    self.state.target_lost_frames += 1
                    print(f"❓ [Approach Phase] Vision Target lost ({self.state.target_lost_frames} frames)")
                    if self.state.target_lost_frames > 20:
                        print("⚠️ Target lost in approach → returning to SCAN")
                        self.state.phase = "scan"
                        self.state.reset_phase_timer()
                        # Reset inputs if target lost
                        yaw_input = 0
                        pitch_input = K_PITCH_P * clamp(self.pitch, -0.5, 0.5) + self.pitch_rate
            else:
                print("⚠️ No camera image available for vision approach.")
                # Optionally switch back to scan if camera fails
                # self.state.phase = "scan"
                # self.state.reset_phase_timer()

        # --- Initiate Landing Transition ---
        if initiate_landing and not self.state.landing_initiated:
            print("🛬 Transitioning to LAND phase.")
            self.state.phase = "land"
            self.state.landing_initiated = True
            self.state.reset_phase_timer()
        elif self.state.phase == "approach": # Apply motors only if still in approach
            self._apply_motor_velocities(
                K_VERTICAL_THRUST + vertical_input - roll_input + pitch_input - yaw_input,
                K_VERTICAL_THRUST + vertical_input + roll_input + pitch_input + yaw_input,
                K_VERTICAL_THRUST + vertical_input - roll_input - pitch_input + yaw_input,
                K_VERTICAL_THRUST + vertical_input + roll_input - pitch_input - yaw_input
            )

    def _run_land_phase(self):
        """Handles the final landing descent."""
        print(f"🛬 [Land Phase] Altitude: {self.altitude:.2f}, Target Alt: {LANDING_TARGET_ALTITUDE}")

        if self.altitude <= LANDING_TARGET_ALTITUDE:
            print("✅ Landed! Stopping motors.")
            self.stop_motors()
            return # Exit phase logic early

        # --- Vertical control for Landing --- 
        # Use consistent PID logic, targetting LANDING_TARGET_ALTITUDE
        alt_error_landing = LANDING_TARGET_ALTITUDE - self.altitude
        # Apply offset and cubic function like other phases
        clamped_diff_alt_landing = clamp(alt_error_landing + K_VERTICAL_OFFSET, -1.0, 1.0) 
        vertical_input = K_VERTICAL_P * (clamped_diff_alt_landing ** 3)
        # Ensure it doesn't try to climb if slightly below target?
        # Clamp output to be non-positive might be safer
        # vertical_input = min(vertical_input, 0.0) # Optional: prevent climbing
        
        # Set base thrust slightly above hover thrust
        landing_thrust = K_VERTICAL_THRUST * LANDING_DESCENT_SPEED

        # --- Horizontal control (base stabilization + Yaw Correction) --- 
        roll_input = K_ROLL_P * clamp(self.roll, -0.5, 0.5) + self.roll_rate
        pitch_input = K_PITCH_P * clamp(self.pitch, -0.5, 0.5) + self.pitch_rate # Base stabilization pitch only
        yaw_input = 0 # Default, correct with GPS if available

        if self.state.target_position is not None:
            # --- REMOVED OFFSET --- Target Original GPS Position ---
            # LAND_BEHIND_DISTANCE = 1.0 
            # original_local_pos = gps_to_local_coordinates(self.gps_values, self.state.target_position)
            # dx_orig, dy_orig, dz_orig = original_local_pos
            # dist_orig = math.sqrt(dx_orig**2 + dy_orig**2)
            # ... (unit vector calculation removed) ...
            # landing_target_local_x = dx_orig - unit_dx * LAND_BEHIND_DISTANCE
            # landing_target_local_y = dy_orig - unit_dy * LAND_BEHIND_DISTANCE
            # landing_target_local_pos = (landing_target_local_x, landing_target_local_y, dz_orig)
            # print(f"🎯 Original local pos: ({dx_orig:.2f}, {dy_orig:.2f}), Offset local pos (Behind): ({landing_target_local_x:.2f}, {landing_target_local_y:.2f})")

            # --- Use Original Position for Control --- 
            target_local_pos = gps_to_local_coordinates(self.gps_values, self.state.target_position) # Use original target
            distance, target_bearing, alt_error = calculate_position_error((0,0,0), target_local_pos) # Use original target
            
            heading_error = target_bearing - self.yaw
            while heading_error > math.pi: heading_error -= 2 * math.pi
            while heading_error < -math.pi: heading_error += 2 * math.pi
            # print(f"🛰️ [Land Phase] GPS Control -> Offset Target. Dist: {distance_to_offset:.2f}, HeadErr: {heading_error:.2f}") 
            print(f"🛰️ [Land Phase] GPS Control -> Original Target. Dist: {distance:.2f}, HeadErr: {heading_error:.2f}") # Updated Print

            # Yaw correction only
            yaw_input = LANDING_HORIZONTAL_P * heading_error
            yaw_input = clamp(yaw_input, -0.1, 0.1) # Keep very smooth clamp
            
            # Ensure pitch uses only base stabilization (already set above)

        else:
            print(f"👁️ [Land Phase] No GPS, using stabilization only.")
            # Ensure pitch_input is also base stabilization if GPS is lost
            pitch_input = K_PITCH_P * clamp(self.pitch, -0.5, 0.5) + self.pitch_rate

        # Apply motor commands for Land phase
        self._apply_motor_velocities(
            landing_thrust + vertical_input - roll_input + pitch_input - yaw_input,
            landing_thrust + vertical_input + roll_input + pitch_input + yaw_input,
            landing_thrust + vertical_input - roll_input - pitch_input + yaw_input,
            landing_thrust + vertical_input + roll_input - pitch_input - yaw_input
        )

    def _check_safety_conditions(self):
        """Checks for unsafe conditions like extreme angles or phase timeouts."""
        # Extreme angles
        if abs(self.roll) > 0.8 or abs(self.pitch) > 0.8:
            print("⚠️ Extreme angles detected - stopping motors")
            self.stop_motors()
            return False # Unsafe
        
        # Phase timeout
        # Simplified check - add specific timeouts per phase if needed
        if (self.state.phase == "scan" and self.state.get_phase_duration() > MAX_SCAN_TIME) or \
           (self.state.phase == "approach" and self.state.get_phase_duration() > MAX_APPROACH_TIME):
            print(f"⚠️ Phase {self.state.phase} timeout - returning to scan")
            self.state.phase = "scan"
            self.state.reset_phase_timer()
            # Note: No explicit return False here, just resets phase

        return True # Safe to continue
        
    def run(self):
        """Main control loop."""
        print("Starting DroneController run loop...")

        # Main loop
        while self.robot.step(self.timestep) != -1:
            try:
                # --- Sensor & Communication --- 
                self._read_sensors()
                self._receive_gps()

                # --- Safety Checks --- 
                if self.state.emergency_stop: # Check if already stopped
                    break 
                if not self._check_safety_conditions():
                    break # Exit loop if safety check fails and stops motors

                # --- Phase Execution --- 
                if self.state.phase == "takeoff":
                    self._run_takeoff_phase()
                elif self.state.phase == "scan":
                    self._run_scan_phase()
                elif self.state.phase == "approach":
                    self._run_approach_phase()
                elif self.state.phase == "land":
                    self._run_land_phase()
                else:
                    print(f"⚠️ Unknown phase: {self.state.phase}. Stopping.")
                    self.stop_motors()
                    break

                # --- OpenCV Debug Window Update --- 
                # Add this back if you uncommented cv2.imshow calls
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #    print("'q' pressed, stopping simulation.")
                #    self.stop_motors()
                #    break

            except Exception as e:
                print(f"❌ Error in main loop: {e}")
                traceback.print_exc()
                self.stop_motors()
                break

        print("DroneController run loop finished.")
        # Optional: Cleanup OpenCV windows
        # cv2.destroyAllWindows()

# Main execution block
if __name__ == "__main__":
    try:
        controller = DroneController()
        controller.run()
    except Exception as e:
        print(f"❌ Failed to initialize or run DroneController: {e}")
        traceback.print_exc()