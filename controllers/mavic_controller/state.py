# controllers/mavic_controller/state.py
import time

class DroneState:
    """Tracks the drone's current operational state and target information."""
    def __init__(self):
        self.phase = "takeoff"  # Current phase: takeoff, scan, approach, land
        self.target_found = False # Whether the vision system currently sees the target
        self.last_target_time = 0 # Timestamp when the target was last seen (vision)
        self.target_lost_frames = 0 # Consecutive frames the vision target has been lost
        self.phase_start_time = time.time() # Timestamp when the current phase started
        self.emergency_stop = False # Flag indicating if emergency stop was triggered
        self.last_gps_update = 0 # Timestamp of the last valid GPS update received
        self.target_position = None  # Last known target (x, y, z) from GPS in world coordinates
        self.use_vision_backup = False # Flag (currently unused) to force vision backup
        self.landing_initiated = False # Flag to ensure landing transition happens only once

    def reset_phase_timer(self):
        """Resets the timer for the current phase and the landing flag."""
        self.phase_start_time = time.time()
        self.landing_initiated = False # Reset landing flag when phase changes

    def get_phase_duration(self):
        """Returns the duration since the current phase started."""
        return time.time() - self.phase_start_time

    def update_target_position(self, position):
        """Updates the target position if enough time has passed since the last update."""
        # This method implicitly handles target_found for GPS
        # If a position is updated, the target is considered "found" via GPS
        current_time = time.time()
        # TODO: Maybe remove the interval check here if the receiver loop handles it?
        # Keeping it for now as a safety throttle
        # TARGET_UPDATE_INTERVAL = 0.5 # seconds between GPS updates (Defined in main controller)
        # if current_time - self.last_gps_update >= TARGET_UPDATE_INTERVAL:
        self.target_position = position
        self.last_gps_update = current_time
        # Reset vision-specific lost frames when GPS is received
        self.target_lost_frames = 0
        print(f"📍 State updated target GPS: {self.target_position}") # Debug in state 