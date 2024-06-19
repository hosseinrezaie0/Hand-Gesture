# Import required libraries
import cv2
import mediapipe as mp


# Create Mediapipe objects
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence = 0.8, min_tracking_confidence = 0.8)
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cam = cv2.Videocamture(0)

while cam.isOpened():
    ret, frame = cam.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    # Convert the frame from BGR (opencv channel) to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hand and keypoints
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        pass

    cv2.imshow('Window', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

cam.release()
cv2.destroyAllWindows()




