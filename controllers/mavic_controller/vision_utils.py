# controllers/mavic_controller/vision_utils.py
import cv2
import numpy as np

# Minimum contour area to consider a valid target
MIN_TARGET_SIZE = 50 
# Threshold for considering target centered (pixels from center)
CENTERING_THRESHOLD = 50 

# Color detection: wide red ranges
RED_HSV_RANGES = [
    ((0,   50,  50), (10,  255, 255)),
    ((170, 50,  50), (180, 255, 255))
]

def detect_red_car(image_bgr):
    """
    Detects the largest red object in the image based on HSV ranges.
    
    Returns (found, cx, cy, area, bbox)
      found: bool
      cx, cy: center of bounding box
      area: contour area
      bbox: (x1, y1, x2, y2)
    """
    if image_bgr is None:
        print("Error: detect_red_car received None image")
        return False, 0, 0, 0, None
        
    try:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

        # Combined mask for all red ranges
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for (lower, upper) in RED_HSV_RANGES:
            lower_np = np.array(lower)
            upper_np = np.array(upper)
            current_mask = cv2.inRange(hsv, lower_np, upper_np)
            mask |= current_mask

        # --- Optional morphology (commented out) ---
        # kernel = np.ones((3,3), np.uint8)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # --- Debug drawing (can be removed or made conditional) ---
        # debug_img = image_bgr.copy()
        # cv2.drawContours(debug_img, contours, -1, (0, 255, 0), 1)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            if area > MIN_TARGET_SIZE:
                x, y, w, h = cv2.boundingRect(largest_contour)
                cx = x + w // 2
                cy = y + h // 2
                
                # --- Debug drawing for largest contour ---
                # cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                # cv2.circle(debug_img, (cx, cy), 5, (255, 0, 0), -1)
                # cv2.putText(debug_img, f"Area: {area:.0f}", (x, y-10), 
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                # cv2.imshow("Original", debug_img)
                
                return True, cx, cy, area, (x, y, x + w, y + h)

        # --- Show debug image even if no target found ---
        # cv2.imshow("Original", debug_img)
        
        # No large enough contour found
        return False, 0, 0, 0, None
        
    except Exception as e:
        print(f"Error in detect_red_car: {e}")
        return False, 0, 0, 0, None 