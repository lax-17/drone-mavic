from controller import Robot, GPS, InertialUnit, Gyro, Motor, Camera
import cv2
import numpy as np
import time


TIME_STEP = 32

# --- TARGET CAR for Recognition (Bottom Cam) ---
TARGET_CAR_NAME = b"targetCar" # <<< CONFIRM THIS EXACTLY MATCHES the car's 'name' field in Webots

# Safety parameters
MAX_SCAN_TIME = 30; MAX_APPROACH_TIME = 45; MAX_LANDING_TIME = 45 # Increased landing timeout slightly
LANDED_ALTITUDE_THRESHOLD = 0.15; ALTITUDE_TOLERANCE = 0.1
EMERGENCY_STOP_AREA_THRESHOLD_FRONT = 0.5 # Stop if target area > 50% in front view

# Control gains
K_VERTICAL_THRUST = 68.0; K_VERTICAL_OFFSET = 0.6; K_VERTICAL_P = 3.0
K_ROLL_P = 15.0; K_PITCH_P = 15.0

# Altitude Targets
TARGET_ALTITUDE = 3.0; LAND_HOLD_ALTITUDE = 1.0


SCAN_YAW_GAIN = 0.3; CENTERING_THRESHOLD = 50; SCAN_LOST_TARGET_YAW = 0.15
# Approach (using OpenCV Front Cam)
APPROACH_YAW_GAIN = 0.05; APPROACH_TILT = -0.15; APPROACH_AREA_SPEED_FACTOR = 0.8
APPROACH_LOST_TARGET_FRAMES = 15
# NEW Landing Trigger: Based on Area AND Vertical Position in Front Cam 
LANDING_TRIGGER_AREA_THRESHOLD = 0.08 
VERTICAL_POSITION_THRESHOLD = 0.8  
# Landing (using Recognition Bottom Cam)
LANDING_CENTERING_P = 0.008; LANDING_MAX_CORRECTION_TILT = 0.05
# LANDING THRUST BASE !!! Lowered to encourage descent
LANDING_THRUST_BASE = 45.0 # <<< WAS 55.0 - TUNE THIS if descent is too slow/fast
LANDING_VERTICAL_P = 8.0;
LAND_LOST_TARGET_FRAMES_TIMEOUT = 60; LANDING_CENTERING_THRESH = 15 # Not currently used, but kept

# --- Vision Parameters for OpenCV (Front Cam) ---
MIN_TARGET_SIZE = 50
# !!! TUNE THESE RANGES FOR FRONT CAMERA VIEW !!!
RED_HSV_RANGES = [((0, 100, 100), (10, 255, 255)), ((165, 100, 100), (180, 255, 255))]


class DroneState: 
    def __init__(self): self.phase="takeoff"; self.target_found_front=False; self.target_found_bottom=False; self.target_lost_frames=0; self.phase_start_time=time.time(); self.emergency_stop=False; self.active_camera_name="camera"
    def reset_phase_timer(self): self.phase_start_time=time.time()
    def get_phase_duration(self): return time.time()-self.phase_start_time
    def should_abort_phase(self): 
        duration = self.get_phase_duration(); timeouts = {"scan":MAX_SCAN_TIME,"approach":MAX_APPROACH_TIME,"land":MAX_LANDING_TIME}
        if self.phase in timeouts and duration > timeouts[self.phase]:
            print(f"{self.phase.upper()} Timeout");
            if self.phase=="land": emergency_stop("Landing Timeout"); return True
            else: return True
        return False


robot = Robot(); state = DroneState()
gps=robot.getDevice("drone_gps"); imu=robot.getDevice("drone_imu"); gyro=robot.getDevice("gyro")
front_camera=robot.getDevice("camera")
bottom_camera = None
try:
    front_camera.enable(TIME_STEP); print("Front cam enabled (OpenCV).")
    dev = robot.getDevice("bottom_camera");
    if dev: bottom_camera=dev; bottom_camera.enable(TIME_STEP); print("Bottom cam init.")
    if bottom_camera and bottom_camera.hasRecognition(): bottom_camera.recognitionEnable(TIME_STEP); print("Bottom cam Recognition enabled.")
    elif bottom_camera: print("⚠Bottom cam has NO Recognition node!")
    else: print(" Bottom cam not found/invalid.")
    
except Exception as e: print(f"Camera init error: {e}")

gps.enable(TIME_STEP); imu.enable(TIME_STEP); gyro.enable(TIME_STEP)
motor_names = ["front left propeller","front right propeller","rear left propeller","rear right propeller"]
motors = [];
try:
    motors = [robot.getDevice(name) for name in motor_names];
    if None in motors: raise ValueError(f"Motors not found: {[motor_names[i] for i,m in enumerate(motors) if m is None]}")
    for m in motors: m.setPosition(float("inf")); m.setVelocity(0.0);
    print("Motors initialized.")
except Exception as e: print(f"MOTOR INIT FAILED: {e}"); state.emergency_stop = True


cv2.namedWindow("Camera View", cv2.WINDOW_NORMAL)


def clamp(v, mn, mx): return max(mn, min(v, mx))
def emergency_stop(reason=""): global state; state.emergency_stop=True; print(f"\n⚠EMERGENCY STOP: {reason}"); [m.setVelocity(0.0) for m in motors if m is not None]
def land_complete(): print("\n Landed Successfully."); emergency_stop("Landed")
def detect_red_target_opencv(img, min_area): # Keep full function
    try:
        if img is None: return False,0,0,0,None
        hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV); mask=np.zeros(hsv.shape[:2],dtype=np.uint8)
        for(l,u) in RED_HSV_RANGES: mask|=cv2.inRange(hsv,np.array(l),np.array(u))
        cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);
        if not cnts: return False,0,0,0,None
        c=max(cnts,key=cv2.contourArea); area=cv2.contourArea(c);
        if area>min_area: x,y,w,h=cv2.boundingRect(c); return True,x+w//2,y+h//2,area,(x,y,x+w,y+h)
        else: return False,0,0,area,None
    except Exception as e: print(f"OpenCV Detect Err: {e}"); return False,0,0,0,None

def check_recognition_bottom(camera): # Keep full function
    if not camera or not camera.hasRecognition() or camera.getRecognitionNumberOfObjects() == 0: return False, 0, 0, (0, 0)
    objects = camera.getRecognitionObjects();
    for obj in objects:

        if obj.getModel() == TARGET_CAR_NAME.decode('utf-8'):
            pos = obj.getPositionOnImage(); size = obj.getSizeOnImage();
            px = int(round(pos[0])); py = int(round(pos[1]))
            sx = int(round(size[0])); sy = int(round(size[1]))
            return True, px, py, (sx, sy)
    return False, 0, 0, (0, 0)

def display_camera_feed(cam_dev, cam_name, found, cx, cy, area_or_size, bbox): # Keep full function
    global state, altitude;
    if cam_dev is None: return
    img_data = cam_dev.getImage(); w = cam_dev.getWidth(); h = cam_dev.getHeight()
    if not img_data or w<=0 or h<=0: return
    try:
        img_array = np.frombuffer(img_data, np.uint8).reshape((h, w, 4))
        bgr_img = cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)
        dbg_img = bgr_img.copy()


        if found:
            if bbox:
                x1,y1,x2,y2 = bbox
                cv2.rectangle(dbg_img, (x1,y1), (x2,y2), (0,0,255), 2) # Red box
                cv2.putText(dbg_img, f"Area:{area_or_size:.0f}", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1) # Green text
                cv2.circle(dbg_img, (cx,cy), 5, (255,0,0), -1) # Blue center dot
            elif isinstance(area_or_size, tuple) and len(area_or_size)==2: # From Recognition (Bottom Cam)
                px_w, px_h = area_or_size
                cx_int, cy_int = int(cx), int(cy)
                x1, y1 = cx_int - px_w // 2, cy_int - px_h // 2
                x2, y2 = cx_int + px_w // 2, cy_int + px_h // 2
                cv2.rectangle(dbg_img, (x1, y1), (x2, y2), (0, 255, 255), 2) # Yellow box for recognition
                cv2.putText(dbg_img, f"Size:{px_w}x{px_h}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1) # Yellow text
                cv2.circle(dbg_img, (cx_int, cy_int), 5, (255, 0, 0), -1) # Blue center dot

        cv2.line(dbg_img,(w//2,h//2-10),(w//2,h//2+10),(0,255,255),1); cv2.line(dbg_img,(w//2-10,h//2),(w//2+10,h//2),(0,255,255),1)
        cv2.putText(dbg_img,f"P:{state.phase} C:{cam_name}",(5,15),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,0),1); cv2.putText(dbg_img,f"Alt:{altitude:.2f}", (w-70, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)
        cv2.imshow("Camera View", dbg_img);
    except Exception as e: print(f"Display Err: {e}")


print("Starting drone (Hybrid + Area/Position Trigger)..."); last_phase = ""; altitude = 0.0
while robot.step(TIME_STEP) != -1 and not state.emergency_stop:
    try:
        roll, pitch, yaw = imu.getRollPitchYaw(); altitude = gps.getValues()[2]
        roll_rate, pitch_rate, yaw_rate = gyro.getValues()

        if abs(roll)>0.8 or abs(pitch)>0.8: emergency_stop("Excessive Roll/Pitch"); continue
        if altitude < -0.2 : emergency_stop("Below Ground"); continue # Slightly more tolerance
        if state.phase != last_phase:
            print(f"\n--- Phase Change: {last_phase.upper()} -> {state.phase.upper()} ---")
            last_phase = state.phase

        if state.should_abort_phase():
             # If timeout occurred in any phase other than landing, go back to scanning
             if state.phase != "land" and not state.emergency_stop:
                 print(f"Phase Timeout -> Reverting to SCAN")
                 state.phase = "scan"
                 state.reset_phase_timer()
                 state.target_lost_frames = 0
             # If landing timed out, emergency_stop is already called by should_abort_phase
             continue

        #  Determine Target Altitude
        # Default to target cruising altitude
        current_target_altitude = TARGET_ALTITUDE
        if state.phase == "land" and not state.target_found_bottom:
            current_target_altitude = LAND_HOLD_ALTITUDE

        # Common PID Calculations
        # Altitude control (always active, but thrust overridden in landing descent)
        clamped_diff_alt = clamp(current_target_altitude - altitude + K_VERTICAL_OFFSET, -1.0, 1.0)
        vertical_input_pid = K_VERTICAL_P * (clamped_diff_alt ** 3)
        # Stabilization
        roll_clamped=clamp(roll,-0.5,0.5); pitch_clamped=clamp(pitch,-0.5,0.5)
        roll_stabilization=K_ROLL_P*roll_clamped+roll_rate; pitch_stabilization=K_PITCH_P*pitch_clamped+pitch_rate

        #Default Input Values
        # Start with altitude hold + stabilization
        vertical_thrust = K_VERTICAL_THRUST + vertical_input_pid
        roll_input = roll_stabilization
        pitch_input = pitch_stabilization
        yaw_input = 0.0 # Yaw is controlled per-phase

        # --- FSM: Camera Selection ---
        current_camera = front_camera
        state.active_camera_name = front_camera.getName()
        
        if state.phase == "land" and bottom_camera:
            current_camera = bottom_camera
            state.active_camera_name = bottom_camera.getName()

        found, cx, cy, area_or_size, bbox = False, 0, 0, 0, None
        relative_area = 0.0
        cam_w = current_camera.getWidth() if current_camera else 0 
        cam_h = current_camera.getHeight() if current_camera else 0

        # Process image based on the active camera
        if current_camera == front_camera and cam_w > 0 and cam_h > 0:
            img_data=current_camera.getImage()
            if img_data:
                bgr_img = cv2.cvtColor(np.frombuffer(img_data, np.uint8).reshape((cam_h, cam_w, 4)), cv2.COLOR_BGRA2BGR)
                found, cx, cy, area, bbox = detect_red_target_opencv(bgr_img, MIN_TARGET_SIZE)
                state.target_found_front = found
                if found: relative_area = area / (cam_w * cam_h) 
            else:
                state.target_found_front = False; found = False
            display_camera_feed(current_camera, state.active_camera_name, found, cx, cy, area if found else 0, bbox)

        elif current_camera == bottom_camera and cam_w > 0 and cam_h > 0:
            found, cx, cy, size_px = check_recognition_bottom(current_camera)
            state.target_found_bottom = found; area_or_size = size_px
            display_camera_feed(current_camera, state.active_camera_name, found, cx, cy, area_or_size, None)

        else:
            state.target_found_front=False; state.target_found_bottom=False; found = False


        # --- Phase Logic ---
        print(f"Phase: {state.phase} | Alt: {altitude:.2f} | Target Alt: {current_target_altitude:.1f} | ", end="") # Common log prefix

        if state.phase == "takeoff":
            print(f"Climbing...", end="\r")
            # Target altitude check
            if altitude >= TARGET_ALTITUDE - ALTITUDE_TOLERANCE:
                print("\nReached target altitude.")
                state.phase="scan"; state.reset_phase_timer(); state.target_lost_frames=0

        elif state.phase == "scan": # Uses OpenCV Front Cam
            if state.target_found_front:
                 state.target_lost_frames=0; off_x=cx-(cam_w//2);
                 print(f"Target Found [F]: OffX:{off_x} | ", end="")
                 # Check if centered enough
                 if abs(off_x) < CENTERING_THRESHOLD:
                     print("Centered -> Approach")
                     state.phase="approach"; state.reset_phase_timer(); state.target_lost_frames=0
                 else: # Not centered, apply yaw correction
                     yaw_input = clamp(-SCAN_YAW_GAIN*off_x/(cam_w//2), -0.5, 0.5)
                     print(f"YawIn:{yaw_input:.2f}", end="\r")
            else: # Target not found
                 state.target_lost_frames+=1;
                 print(f"Searching... Lost Frames:{state.target_lost_frames} | ", end="")
                 # Rotate to search if lost for a few frames
                 if state.target_lost_frames > 5: # Start rotating after 5 frames lost
                     yaw_input = SCAN_LOST_TARGET_YAW;
                     print(f"Rotating:{yaw_input:.2f}", end="\r")
                 else: # Wait briefly before rotating
                     print("Holding...", end="\r")


        elif state.phase == "approach": # Uses OpenCV Front Cam
            if state.target_found_front and cam_h > 0 and cam_w > 0: # Ensure cam dims valid
                state.target_lost_frames=0; off_x=cx-(cam_w//2);
                # Check landing trigger conditions
                is_close_enough_area = relative_area > LANDING_TRIGGER_AREA_THRESHOLD
                is_low_in_view = cy > cam_h * VERTICAL_POSITION_THRESHOLD 

                print(f"Target Found [F]: Area:{relative_area:.3f} (>{LANDING_TRIGGER_AREA_THRESHOLD:.3f}?) | "
                      f"VPos:{cy/cam_h:.2f} (>{VERTICAL_POSITION_THRESHOLD:.1f}?) | OffX:{off_x} -> ", end="")

            
                if is_close_enough_area and is_low_in_view:
                    print("CLOSE & LOW -> LAND")
                    if bottom_camera and bottom_camera.hasRecognition():
                        state.phase="land"; state.reset_phase_timer(); state.target_lost_frames=0;
                        continue 
                    else:
                        print("! No Bottom Cam w/ Recog for Landing!"); emergency_stop("No Btm Cam for Land"); continue
                elif relative_area > EMERGENCY_STOP_AREA_THRESHOLD_FRONT:
                    print("! STOP (Too Close)"); emergency_stop(f"Front Area Exceeded {relative_area:.2%}"); continue
                else:
                    print("Approaching...")
                    speed_scale = relative_area / LANDING_TRIGGER_AREA_THRESHOLD if LANDING_TRIGGER_AREA_THRESHOLD > 0 else 1.0
                    approach_factor = 1.0 - min(speed_scale, APPROACH_AREA_SPEED_FACTOR) 
                    forward_tilt = APPROACH_TILT * approach_factor
                    # Apply pitch for forward movement and yaw for centering
                    pitch_input += K_PITCH_P * clamp(forward_tilt, -0.4, 0.0)
                    yaw_input = clamp(-APPROACH_YAW_GAIN * off_x / (cam_w // 2), -0.3, 0.3) 
                    print(f" Tilt:{forward_tilt:.2f} YawIn:{yaw_input:.2f}", end="\r")

            else:
                state.target_lost_frames += 1;
                print(f"Target Lost ({state.target_lost_frames}/{APPROACH_LOST_TARGET_FRAMES})", end="\r")
                if state.target_lost_frames > APPROACH_LOST_TARGET_FRAMES:
                    print("\n! Lost target during approach -> Reverting to SCAN")
                    state.phase = "scan"; state.reset_phase_timer()

        elif state.phase == "land":
            if not bottom_camera:
                print("! LANDING ABORTED - No Bottom Cam -> SCAN"); state.phase = "scan"; state.reset_phase_timer(); continue

            if state.target_found_bottom and cam_w > 0 and cam_h > 0: # Target Found via Recognition
                state.target_lost_frames = 0
                
                #Calculate Descent Thrust 
                calculated_thrust = LANDING_THRUST_BASE + LANDING_VERTICAL_P * altitude
                vertical_thrust = clamp(calculated_thrust, 0, K_VERTICAL_THRUST * 1.1) # Descent Thrust
                #Calculate Centering Corrections 
                offset_x = cx - cam_w // 2; offset_y = cy - cam_h // 2
                roll_correction = clamp(LANDING_CENTERING_P * offset_x, -LANDING_MAX_CORRECTION_TILT, LANDING_MAX_CORRECTION_TILT)
                
                # Check pitch sign: If target Y > center Y (target lower on screen), want positive pitch offset? Depends on IMU frame. Assuming (-) makes drone pitch *down* towards target.
                pitch_correction = clamp(-LANDING_CENTERING_P * offset_y, -LANDING_MAX_CORRECTION_TILT, LANDING_MAX_CORRECTION_TILT) # Adjust sign if needed based on testing
                # --- Apply Corrections ---
                roll_input = roll_stabilization + K_ROLL_P * roll_correction # Add centering correction
                pitch_input = pitch_stabilization + K_PITCH_P * pitch_correction # Add centering correction
                yaw_input = 0 # No yaw during final descent

                print(f"Target Found [R]: Descending... Off(X:{offset_x},Y:{offset_y}) | "
                      f"Thrust Calc:{calculated_thrust:.1f}->{vertical_thrust:.1f}", end="\r") 

                if altitude < LANDED_ALTITUDE_THRESHOLD:
                    land_complete(); continue

            else: 
                state.target_lost_frames += 1
                roll_input = roll_stabilization; pitch_input = pitch_stabilization; yaw_input = 0

                print(f"Target Lost [R]({state.target_lost_frames}/{LAND_LOST_TARGET_FRAMES_TIMEOUT}): Holding Alt @ {LAND_HOLD_ALTITUDE:.1f}m", end="\r")

                # If lost for too long during landing, abort and rescan
                if state.target_lost_frames > LAND_LOST_TARGET_FRAMES_TIMEOUT:
                    print("\n! Lost Target too long during landing -> Reverting to SCAN")
                    state.phase = "scan"; state.reset_phase_timer(); continue


        if not state.emergency_stop:
            m1_speed = vertical_thrust - roll_input + pitch_input - yaw_input # Front Left
            m2_speed = vertical_thrust + roll_input + pitch_input + yaw_input # Front Right
            m3_speed = vertical_thrust - roll_input - pitch_input + yaw_input # Rear Left
            m4_speed = vertical_thrust + roll_input - pitch_input - yaw_input # Rear Right

            if motors and all(motors):
                motors[0].setVelocity(m1_speed)    # Front Left
                motors[1].setVelocity(-m2_speed)   # Front Right (Often reversed)
                motors[2].setVelocity(-m3_speed)   # Rear Left   (Often reversed)
                motors[3].setVelocity(m4_speed)    # Rear Right
            elif not state.emergency_stop: # Prevent error spam after stop
                 print("ERROR: Motors invalid or not initialized!")
                 emergency_stop("Motor Failure")

        # OpenCV Update
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print("\nUser requested exit.")
            emergency_stop("User Exit")
            break

    except Exception as e:
        print(f"\n❌ MAIN LOOP ERROR: {e} ❌")
        import traceback
        traceback.print_exc()
        emergency_stop("Loop Exception")
        break


print("\nSimulation loop finished or stopped.")
if not state.emergency_stop: 
    emergency_stop("Script End")

cv2.destroyAllWindows()
print("OpenCV windows closed.")

print("Script finished.")