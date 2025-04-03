from controller import Robot, GPS, Emitter
import numpy as np

TIME_STEP = 32

# Initialize robot and devices
robot = Robot()
gps = robot.getDevice("gps")
emitter = robot.getDevice("emitter")

if gps is None:
    print("⚠️ ERROR: GPS device not found on car!")
    exit(1)

if emitter is None:
    print("⚠️ ERROR: Emitter device not found on car!")
    exit(1)

# Enable GPS
gps.enable(TIME_STEP)

print("✅ Car controller initialized")
print(f"📡 Emitter channel: {emitter.getChannel()}")

# Main loop
while robot.step(TIME_STEP) != -1:
    # Get GPS values
    gps_list = gps.getValues()
    gps_values = np.array(gps_list, dtype=np.float64)
    
    # Send GPS data
    try:
        emitter.send(gps_values.tobytes())
        print(f"📤 Sent GPS: {gps_values}")
    except Exception as e:
        print(f"⚠️ Error sending GPS data: {e}") 