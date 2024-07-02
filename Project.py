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
        
def count_fingers(result, frame, dir):
    for hand_landmark in result.multi_hand_landmarks:
        mp_drawing.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)
        # Count fingers for Up direction
        if dir == "Up":
            fingers = []
            for idx, landmark in enumerate(hand_landmark.landmark):
                if idx == 4:  # Thumb
                    if (landmark.x < hand_landmark.landmark[3].x):
                        fingers.append(1)
                    else:
                        fingers.append(0)
                if idx == 8:  # Index
                    if (landmark.y < hand_landmark.landmark[6].y):
                        fingers.append(1)
                    else:
                        fingers.append(0)
                if idx == 12:  # Middle
                    if (landmark.y < hand_landmark.landmark[10].y):
                        fingers.append(1)
                    else:
                        fingers.append(0)
                if idx == 16:  # Ring
                    if (landmark.y < hand_landmark.landmark[14].y):
                        fingers.append(1)
                    else:
                        fingers.append(0)
                if idx == 20:  # Pinky
                    if (landmark.y < hand_landmark.landmark[18].y):
                        fingers.append(1)
                    else:
                        fingers.append(0)

        # Count fingers for down direction
        if dir == "Down":
            fingers = []
            for index, landmark in enumerate(hand_landmark.landmark):
                if index == 4:  # Thumb
                    if (landmark.x > hand_landmark.landmark[3].x):
                        fingers.append(1)
                    else:
                        fingers.append(0)
                    
                if index == 8:  # Index 
                    if (landmark.y > hand_landmark.landmark[6].y):
                        fingers.append(1)
                    else:
                        fingers.append(0)
                
                if index == 12:  # Middle
                    if (landmark.y > hand_landmark.landmark[10].y):
                        fingers.append(1)
                    else:
                        fingers.append(0)
                
                if index == 16:  # Ring
                    if (landmark.y > hand_landmark.landmark[14].y):
                        fingers.append(1)
                    else:
                        fingers.append(0)

                if index == 20:  # Pinky
                    if (landmark.y > hand_landmark.landmark[18].y):
                        fingers.append(1)
                    else:
                        fingers.append(0)
                # Count fingers for down direction
        
        # Count fingers for right direction
        if dir == "Right":
            fingers = []
            for idx, landmark in enumerate(hand_landmark.landmark):
                if idx == 4:  # Thumb
                    if (landmark.y < hand_landmark.landmark[3].y):
                        fingers.append(1)
                    else:
                        fingers.append(0)
                if idx == 8:  # Index
                    if (landmark.x > hand_landmark.landmark[6].x):
                        fingers.append(1)
                    else:
                        fingers.append(0)
                if idx == 12:  # Middle
                    if (landmark.x > hand_landmark.landmark[10].x):
                        fingers.append(1)
                    else:
                        fingers.append(0)
                if idx == 16:  # Ring
                    if (landmark.x > hand_landmark.landmark[14].x):
                        fingers.append(1)
                    else:
                        fingers.append(0)
                if idx == 20:  # Pinky
                    if (landmark.x > hand_landmark.landmark[18].x):
                        fingers.append(1)
                    else:
                        fingers.append(0)

        # Count fingers for left direction
        if dir == "Left":
            fingers = []
            for idx, landmark in enumerate(hand_landmark.landmark):
                if idx == 4:  # Thumb
                    if (landmark.y > hand_landmark.landmark[3].y):
                        fingers.append(1)
                    else:
                        fingers.append(0)
                if idx == 8:  # Index
                    if (landmark.x < hand_landmark.landmark[6].x):
                        fingers.append(1)
                    else:
                        fingers.append(0)
                if idx == 12:  # Middle
                    if (landmark.x < hand_landmark.landmark[10].x):
                        fingers.append(1)
                    else:
                        fingers.append(0)
                if idx == 16:  # Ring
                    if (landmark.x < hand_landmark.landmark[14].x):
                        fingers.append(1)
                    else:
                        fingers.append(0)
                if idx == 20:  # Pinky
                    if (landmark.x < hand_landmark.landmark[18].x):
                        fingers.append(1)
                    else:
                        fingers.append(0)


        finger_count = fingers.count(1)
        return finger_count

def left_or_right(result, dir):
    for hand_landmark in result.multi_hand_landmarks:
        if dir == "Up":
            for idx, landmark in enumerate(hand_landmark.landmark):
                if idx == 4:
                    thumb_tip_x = landmark.x
                if idx == 20:
                    pinky_tip_x = landmark.x
            if thumb_tip_x > pinky_tip_x:
                return "Left Hand"
            else:
                return "Right Hand"
        if dir == "Down":
            for idx, landmark in enumerate(hand_landmark.landmark):
                if idx == 4:
                    thumb_tip_x = landmark.x
                if idx == 20:
                    pinky_tip_x = landmark.x
            if thumb_tip_x > pinky_tip_x:
                return "Right Hand"
            else:
                return "Left Hand"
        if dir == "Right":
            for idx, landmark in enumerate(hand_landmark.landmark):
                if idx == 4:
                    thumb_tip_y = landmark.y
                if idx == 20:
                    pinky_tip_y = landmark.y
            if thumb_tip_y > pinky_tip_y:
                return "Left Hand"
            else:
                return "Right Hand"
        if dir == "Left":
            for idx, landmark in enumerate(hand_landmark.landmark):
                if idx == 4:
                    thumb_tip_y = landmark.y
                if idx == 20:
                    pinky_tip_y = landmark.y
            if thumb_tip_y > pinky_tip_y:
                return "Right Hand"
            else:
                return "Left Hand"

# Create Mediapipe objects
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence = 0.8, min_tracking_confidence = 0.8)
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cam = cv2.VideoCapture(0)

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
        dir = get_direction(result, rgb_frame)
        count = count_fingers(result, rgb_frame, dir)
        label = left_or_right(result, dir)
        cv2.putText(frame, f"Number: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Direction: {dir}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        # cv2.putText(frame, f"Label : {label}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)

    cv2.imshow('Window', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

cam.release()
cv2.destroyAllWindows()




