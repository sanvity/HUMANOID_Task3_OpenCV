import cv2
import numpy as np
from collections import deque

blue_lower = np.array([100, 150, 0])
blue_upper = np.array([140, 255, 255])
skin_lower = np.array([0, 20, 70])
skin_upper = np.array([20, 255, 255])

def is_bottle_upright(contour):
    rect = cv2.minAreaRect(contour)
    width, height = rect[1]
    aspect_ratio = height / width 
    return aspect_ratio > 2.5

def detect_bottle_flip(video_path):
    cap = cv2.VideoCapture(video_path)
    blue_positions = deque(maxlen=30)
    bottom_positions = deque(maxlen=30)
    hand_detected = False
    flip_detected = False
    upright_landed = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        skin_mask = cv2.inRange(hsv, skin_lower, skin_upper)
        
        blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        skin_contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if blue_contours:
            largest_blue_contour = max(blue_contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(largest_blue_contour)
            center = (int(x), int(y))
            radius = int(radius)
            blue_positions.append(center)
            cv2.circle(frame, center, radius, (255, 0, 0), 2) #blue top
            
            if radius > 10 and is_bottle_upright(largest_blue_contour):
                upright_landed = True

            bottle_height = 6 * radius
            bottom_center = (center[0], center[1] + bottle_height)
            bottom_positions.append(bottom_center)
            cv2.circle(frame, bottom_center, radius, (0, 255, 0), 2) #green bottom
   
        for contour in skin_contours:
            if cv2.pointPolygonTest(contour, center, False) >= 0:
                hand_detected = True
                break
      
        if len(blue_positions) >= 10 and len(bottom_positions) >= 10:
            dx = blue_positions[-1][0] - blue_positions[0][0]
            dy = blue_positions[-1][1] - blue_positions[0][1]
            vertical_movement = [pos[1] for pos in blue_positions]
            min_y = min(vertical_movement)
            max_y = max(vertical_movement)
            
            bottom_dx = bottom_positions[-1][0] - bottom_positions[0][0]
            bottom_dy = bottom_positions[-1][1] - bottom_positions[0][1]
            bottom_vertical_movement = [pos[1] for pos in bottom_positions]
            bottom_min_y = min(bottom_vertical_movement)
            bottom_max_y = max(bottom_vertical_movement)
            
            if abs(dx) > 50 and abs(dy) > 50 and (max_y - min_y) > 60 and abs(bottom_dx) > 50 and abs(bottom_dy) > 50 and (bottom_max_y - bottom_min_y) > 60:
                flip_detected = True

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    if flip_detected and upright_landed and not hand_detected:#and upright_landed:
        return "Bottle flip successful"
    else:
        return "Bottle flip failed"

#video_path = '/Users/sanvijain/LearningOpencv/bottleflip.mov'
#video_path = '/Users/sanvijain/LearningOpencv/NoFlip.mov'
video_path=0

result = detect_bottle_flip(video_path)
print(result)
