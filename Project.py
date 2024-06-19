# Import required libraries
import cv2
import mediapipe as mp

def get_direction(result, frame):
    for hand_landmarks in result.multi_hand_landmarks:
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Get the keypoints of wrist and middel finger
        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        middel_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

        # Calculate the direction vector
        direction_vector = (middel_finger_tip.x - wrist.x, middel_finger_tip.y - wrist.y)

        # Detect direction
        x, y = direction_vector
        if abs(x) > abs(y):
            if x > 0:
                return "Right"
            else:
                return "Left"
        else:
            if y > 0:
                return "Down"
            else:
                return "Up"
        

        
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




