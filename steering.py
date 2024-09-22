import cv2
import mediapipe as mp
from pynput.keyboard import Key, Controller
import math

# Initialize MediaPipe Hands and drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize keyboard controller
keyboard = Controller()

# Function to detect gestures and control steering
def detect_gestures(image, results):
    if not results.multi_hand_landmarks or len(results.multi_hand_landmarks) < 2:
        return image

    image_height, image_width, _ = image.shape
    hand_landmarks = results.multi_hand_landmarks

    # Get the wrist coordinates of both hands
    wrist1 = hand_landmarks[0].landmark[mp_hands.HandLandmark.WRIST]
    wrist2 = hand_landmarks[1].landmark[mp_hands.HandLandmark.WRIST]

    wrist1_coords = (int(wrist1.x * image_width), int(wrist1.y * image_height))
    wrist2_coords = (int(wrist2.x * image_width), int(wrist2.y * image_height))

    cv2.circle(image, wrist1_coords, 10, (0, 255, 0), -1)
    cv2.circle(image, wrist2_coords, 10, (0, 255, 0), -1)
    cv2.line(image, wrist1_coords, wrist2_coords, (0, 255, 0), 2)

    # Calculate the slope of the line between the wrists
    delta_x = wrist2_coords[0] - wrist1_coords[0]
    delta_y = wrist2_coords[1] - wrist1_coords[1]

    if delta_x == 0:
        angle = 90
    else:
        angle = math.atan2(delta_y, delta_x) * 180 / math.pi

    # Determine steering direction based on the angle
    if angle < -10:
        # Turn left
        print("Turn left")
        keyboard.release('s')
        keyboard.release('d')
        keyboard.press('a')
    elif angle > 10:
        # Turn right
        print("Turn right")
        keyboard.release('s')
        keyboard.release('a')
        keyboard.press('d')
    else:
        # Move forward
        print("Move forward")
        keyboard.release('a')
        keyboard.release('d')
        keyboard.press('w')

    return image

# Start webcam feed
cap = cv2.VideoCapture(0)
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        frame = detect_gestures(frame, results)

        # Display the resulting frame
        cv2.imshow('Virtual Steering', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
