import math

def clamp(value, min_val, max_val):
    """Clamps a value between a minimum and maximum."""
    return max(min_val, min(value, max_val))

def calculate_position_error(current_pos, target_pos):
    """Calculate distance, bearing to target, and altitude difference."""
    # Assumes current_pos is relative origin (0,0,0) for drone control
    dx = target_pos[0] - current_pos[0]
    dy = target_pos[1] - current_pos[1]
    dz = target_pos[2] - current_pos[2]
    
    distance = math.sqrt(dx*dx + dy*dy)
    # target_bearing is the angle in the world XY plane from drone's local X-axis to the target
    target_bearing = math.atan2(dy, dx)
    
    return distance, target_bearing, dz

def gps_to_local_coordinates(gps_values, target_gps):
    """
    Convert absolute target GPS coordinates to local coordinates relative to the drone.
    Assumes drone is at the origin of its local frame.
    NOTE: This is a simplified conversion assuming aligned axes and neglecting curvature.
          For more accuracy, proper coordinate frame transformations (e.g., ENU/NED) are needed.
    """
    if gps_values is None or target_gps is None:
        print("Error: gps_to_local_coordinates received None input")
        return (0, 0, 0) # Or raise error
        
    # Simple subtraction assumes drone's local frame is aligned with world frame at its position
    dx = target_gps[0] - gps_values[0]
    dy = target_gps[1] - gps_values[1]
    # dz = target_gps[2] - gps_values[2] # Altitude difference is often handled separately
    # Return relative position in drone's XY plane (z can be handled by altitude control)
    # We return target Z relative to drone Z separately if needed, or handle via altitude PID
    local_z = target_gps[2] - gps_values[2] # Relative altitude
    
    # The main controller uses calculate_position_error with (0,0,0) as current pos
    # So this function essentially provides the target's position in a frame relative to the drone's current GPS coords.
    return (dx, dy, local_z) # Return relative X, Y, Z 