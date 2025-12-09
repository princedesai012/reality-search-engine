import cv2

# Try to open the default camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    print("Camera opened successfully!")
    ret, frame = cap.read()
    if ret:
        print("Successfully read a frame from the camera!")
        # Try to show the frame
        try:
            cv2.imshow('Test Window', frame)
            print("Displayed test window. Press any key to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            print("Test completed successfully!")
        except Exception as e:
            print(f"Error displaying window: {e}")
    else:
        print("Failed to read frame from camera.")
    cap.release()
