import cv2
import numpy as np
import os
from datetime import datetime

def setup_directories():
    """Create necessary directories if they don't exist."""
    if not os.path.exists('frames_db'):
        os.makedirs('frames_db')

class MotionDetector:
    def __init__(self, threshold=25, min_area=1000):
        self.threshold = threshold
        self.min_area = min_area
        self.prev_frame = None
        
    def detect_motion(self, current_frame):
        """Detect motion by comparing current frame with previous frame."""
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            return None
            
        frame_delta = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(frame_delta, self.threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        self.prev_frame = gray
        
        # Find contours of the thresholded image
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_rects = []
        for contour in contours:
            if cv2.contourArea(contour) > self.min_area:
                (x, y, w, h) = cv2.boundingRect(contour)
                motion_rects.append((x, y, x + w, y + h))
                
        return motion_rects

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    
    # Set camera resolution (adjust based on your webcam's capabilities)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Initialize motion detector
    motion_detector = MotionDetector()
    
    # Setup directories
    setup_directories()
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Resize frame for faster processing
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        
        # Detect motion
        motion_rects = motion_detector.detect_motion(frame)
        
        # Draw rectangles around motion areas
        if motion_rects:
            for (x1, y1, x2, y2) in motion_rects:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Save frame when motion is detected
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"frames_db/motion_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Motion detected! Saved as {filename}")
        
        # Display the resulting frame
        cv2.imshow('CCTV Motion Detection', frame)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
