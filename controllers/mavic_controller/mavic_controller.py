from controller import Robot, GPS, InertialUnit, Gyro, Motor
import cv2
import numpy as np

# Constants (from your working PID-based controller)
K_VERTICAL_THRUST = 68.5
K_VERTICAL_OFFSET = 0.6
K_VERTICAL_P = 3.0
K_ROLL_P = 50.0
K_PITCH_P = 30.0
TIME_STEP = 32
TARGET_ALTITUDE = 1.0

# Initialize robot and devices
robot = Robot()

gps = robot.getDevice("drone_gps")
gps.enable(TIME_STEP)

imu = robot.getDevice("drone_imu")
imu.enable(TIME_STEP)

gyro = robot.getDevice("gyro")
gyro.enable(TIME_STEP)

# Initialize camera for scanning
camera = robot.getDevice("camera")
camera.enable(TIME_STEP)

motor_names = [
    "front left propeller",
    "front right propeller",
    "rear left propeller",
    "rear right propeller"
]
motors = [robot.getDevice(name) for name in motor_names]
for motor in motors:
    motor.setPosition(float("inf"))
    motor.setVelocity(0.0)

print("[INFO] Python drone controller (PID-based with scanning) initialized.")

# Define the phase: "takeoff", "scan", or "approach"
phase = "takeoff"

while robot.step(TIME_STEP) != -1:
    # Get sensor readings
    roll, pitch, yaw = imu.getRollPitchYaw()
    altitude = gps.getValues()[2]  # Z is the vertical axis
    roll_rate, pitch_rate, yaw_rate = gyro.getValues()

    # Process keyboard input for disturbances and altitude adjustments
    roll_disturbance = 0.0
    pitch_disturbance = 0.0
    yaw_disturbance = 0.0

    # --- Phase: TAKEOFF ---
    if phase == "takeoff":
        if altitude < TARGET_ALTITUDE - 0.1:
            # PID for altitude stabilization during takeoff
            clamped_diff_alt = max(min(TARGET_ALTITUDE - altitude + K_VERTICAL_OFFSET, 1.0), -1.0)
            vertical_input = K_VERTICAL_P * (clamped_diff_alt ** 3)
            roll_input = K_ROLL_P * max(min(roll, 1.0), -1.0) + roll_rate + roll_disturbance
            pitch_input = K_PITCH_P * max(min(pitch, 1.0), -1.0) + pitch_rate + pitch_disturbance

            front_left = K_VERTICAL_THRUST + vertical_input - roll_input + pitch_input - yaw_disturbance
            front_right = K_VERTICAL_THRUST + vertical_input + roll_input + pitch_input + yaw_disturbance
            rear_left = K_VERTICAL_THRUST + vertical_input - roll_input - pitch_input + yaw_disturbance
            rear_right = K_VERTICAL_THRUST + vertical_input + roll_input - pitch_input - yaw_disturbance

            motors[0].setVelocity(front_left)
            motors[1].setVelocity(-front_right)
            motors[2].setVelocity(-rear_left)
            motors[3].setVelocity(rear_right)

            print(f"[TAKEOFF] Altitude: {altitude:.2f}")
            continue  # Skip the rest until target altitude is reached
        else:
            print("🟢 Altitude reached. Switching to SCAN phase.")
            phase = "scan"

    # --- Phase: SCAN ---
    if phase == "scan":
        # Maintain stability using PID
        clamped_diff_alt = max(min(TARGET_ALTITUDE - altitude + K_VERTICAL_OFFSET, 1.0), -1.0)
        vertical_input = K_VERTICAL_P * (clamped_diff_alt ** 3)
        roll_input = K_ROLL_P * max(min(roll, 1.0), -1.0) + roll_rate + roll_disturbance
        pitch_input = K_PITCH_P * max(min(pitch, 1.0), -1.0) + pitch_rate + pitch_disturbance

        # Introduce a constant yaw rotation for scanning
        yaw_rotation = 0.5  # adjust as needed for scanning speed
        yaw_input = yaw_rotation + yaw_disturbance

        front_left = K_VERTICAL_THRUST + vertical_input - roll_input + pitch_input - yaw_input
        front_right = K_VERTICAL_THRUST + vertical_input + roll_input + pitch_input + yaw_input
        rear_left = K_VERTICAL_THRUST + vertical_input - roll_input - pitch_input + yaw_input
        rear_right = K_VERTICAL_THRUST + vertical_input + roll_input - pitch_input - yaw_input

        motors[0].setVelocity(front_left)
        motors[1].setVelocity(-front_right)
        motors[2].setVelocity(-rear_left)
        motors[3].setVelocity(rear_right)

        # Process the camera image to search for a red object
        img = camera.getImage()
        if img:
            width = camera.getWidth()
            height = camera.getHeight()
            img_array = np.frombuffer(img, np.uint8).reshape((height, width, 4))
            bgr_img = cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)
            hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

            # Define thresholds for red detection (two ranges)
            lower_red1 = np.array([0, 80, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 80, 50])
            upper_red2 = np.array([180, 255, 255])
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = mask1 | mask2

            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    offset = cx - width // 2
                    if abs(offset) < 30:
                        print("🎯 Red object centered. Switching to APPROACH phase.")
                        phase = "approach"
                    else:
                        print("🔄 Red object detected but not centered.")
            # Optionally display the camera view (if supported)
            cv2.imshow("Drone Camera View", bgr_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # --- Phase: APPROACH ---
    if phase == "approach":
        # For approaching, we tilt forward by modifying the pitch
        forward_tilt = -0.1  # adjust to control forward speed
        effective_pitch = pitch + forward_tilt

        clamped_diff_alt = max(min(TARGET_ALTITUDE - altitude + K_VERTICAL_OFFSET, 1.0), -1.0)
        vertical_input = K_VERTICAL_P * (clamped_diff_alt ** 3)
        roll_input = K_ROLL_P * max(min(roll, 1.0), -1.0) + roll_rate + roll_disturbance
        pitch_input = K_PITCH_P * max(min(effective_pitch, 1.0), -1.0) + pitch_rate + pitch_disturbance

        front_left = K_VERTICAL_THRUST + vertical_input - roll_input + pitch_input - yaw_disturbance
        front_right = K_VERTICAL_THRUST + vertical_input + roll_input + pitch_input + yaw_disturbance
        rear_left = K_VERTICAL_THRUST + vertical_input - roll_input - pitch_input + yaw_disturbance
        rear_right = K_VERTICAL_THRUST + vertical_input + roll_input - pitch_input - yaw_disturbance

        motors[0].setVelocity(front_left)
        motors[1].setVelocity(-front_right)
        motors[2].setVelocity(-rear_left)
        motors[3].setVelocity(rear_right)
        print("Approaching the target...")

    # Debug output for monitoring
    print(f"[Phase: {phase}] [Altitude: {altitude:.2f}] [Roll: {roll:.2f}] [Pitch: {pitch:.2f}]")

# Clean up the OpenCV windows when exiting
cv2.destroyAllWindows()
