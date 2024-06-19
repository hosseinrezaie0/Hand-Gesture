# Import required libraries
import cv2
import mediapipe as mp


# Create Mediapipe objects
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence = 0.8, min_tracking_confidence = 0.8)
mp_drawing = mp.solutions.drawing_utils




