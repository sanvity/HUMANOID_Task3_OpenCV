import cv2
import numpy as np

cap = cv2.VideoCapture('/Users/sanvijain/LearningOpencv/target.mov')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("end")
        break
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    circular_contours = []
    for contour in contours:
        ((x, y), radius) = cv2.minEnclosingCircle(contour) 
        circle_area = np.pi * (radius ** 2)
        contour_area = cv2.contourArea(contour)
       
        if contour_area / circle_area > 0.7:  
            circular_contours.append(contour)
    
    
    if circular_contours:
        largest_contour = max(contours, key=cv2.contourArea)
        
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:  # area not 0
            cX = int(M["m10"] / M["m00"])  #centroid
            cY = int(M["m01"] / M["m00"])
            
            cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 4)
            cv2.circle(frame, (cX, cY), 7, (255, 255, 255), -1)
            cv2.putText(frame, "shoot!", (cX - 20, cY - 20),cv2.FONT_ITALIC, 0.5, (255, 0, 0), 2)
            print("Shoot")
    else:
        print("No circles found")
    
    cv2.imshow('Frame', frame)
   # cv2.imshow('Mask', mask)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
