from controller import Robot, GPS, InertialUnit, Gyro, Motor, Keyboard

# Constants (based on original C code logic)
K_VERTICAL_THRUST = 68.5
K_VERTICAL_OFFSET = 0.6
K_VERTICAL_P = 3.0
K_ROLL_P = 50.0
K_PITCH_P = 30.0
TIME_STEP = 32
TARGET_ALTITUDE = 1.0

# Initialize robot and devices
robot = Robot()
keyboard = Keyboard()
keyboard.enable(TIME_STEP)

gps = robot.getDevice("drone_gps")
gps.enable(TIME_STEP)

imu = robot.getDevice("drone_imu")
imu.enable(TIME_STEP)

gyro = robot.getDevice("gyro")
gyro.enable(TIME_STEP)

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

print("[INFO] Python drone controller (PID-based) initialized.")

# Main loop
while robot.step(TIME_STEP) != -1:
    roll, pitch, _ = imu.getRollPitchYaw()
    altitude = gps.getValues()[2]  # Z is vertical axis in Mavic 2 Pro
    roll_rate, pitch_rate, _ = gyro.getValues()

    # Default disturbances from keyboard
    roll_disturbance = 0.0
    pitch_disturbance = 0.0
    yaw_disturbance = 0.0

    # Keyboard control
    key = keyboard.getKey()
    while key > 0:
        if key == Keyboard.UP:
            pitch_disturbance = -2.0
        elif key == Keyboard.DOWN:
            pitch_disturbance = 2.0
        elif key == Keyboard.LEFT:
            yaw_disturbance = 1.3
        elif key == Keyboard.RIGHT:
            yaw_disturbance = -1.3
        elif key == (Keyboard.SHIFT + Keyboard.LEFT):
            roll_disturbance = 1.0
        elif key == (Keyboard.SHIFT + Keyboard.RIGHT):
            roll_disturbance = -1.0
        elif key == (Keyboard.SHIFT + Keyboard.UP):
            TARGET_ALTITUDE += 0.05
            print(f"[DEBUG] Target Altitude increased to {TARGET_ALTITUDE:.2f}")
        elif key == (Keyboard.SHIFT + Keyboard.DOWN):
            TARGET_ALTITUDE -= 0.05
            print(f"[DEBUG] Target Altitude decreased to {TARGET_ALTITUDE:.2f}")
        key = keyboard.getKey()

    # PID computations
    roll_input = K_ROLL_P * max(min(roll, 1.0), -1.0) + roll_rate + roll_disturbance
    pitch_input = K_PITCH_P * max(min(pitch, 1.0), -1.0) + pitch_rate + pitch_disturbance
    clamped_diff_alt = max(min(TARGET_ALTITUDE - altitude + K_VERTICAL_OFFSET, 1.0), -1.0)
    vertical_input = K_VERTICAL_P * (clamped_diff_alt ** 3)

    # Final motor commands
    front_left = K_VERTICAL_THRUST + vertical_input - roll_input + pitch_input - yaw_disturbance
    front_right = K_VERTICAL_THRUST + vertical_input + roll_input + pitch_input + yaw_disturbance
    rear_left = K_VERTICAL_THRUST + vertical_input - roll_input - pitch_input + yaw_disturbance
    rear_right = K_VERTICAL_THRUST + vertical_input + roll_input - pitch_input - yaw_disturbance

    motors[0].setVelocity(front_left)
    motors[1].setVelocity(-front_right)
    motors[2].setVelocity(-rear_left)
    motors[3].setVelocity(rear_right)

    # Debug output
    print(f"[Z: {altitude:.2f}] [TGT: {TARGET_ALTITUDE:.2f}] [R: {roll:.2f}] [P: {pitch:.2f}]")

