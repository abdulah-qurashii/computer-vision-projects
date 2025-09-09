import cv2
import mediapipe as mp
import pyautogui

# Setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)

screen_w, screen_h = pyautogui.size()

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)  # mirror image
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Index finger tip coordinates (landmark 8)
            x = int(hand_landmarks.landmark[8].x * screen_w)
            y = int(hand_landmarks.landmark[8].y * screen_h)

            pyautogui.moveTo(x, y)  # move mouse with index finger

            # Thumb tip (landmark 4) - to check click
            thumb_x = int(hand_landmarks.landmark[4].x * screen_w)
            thumb_y = int(hand_landmarks.landmark[4].y * screen_h)

            # If index and thumb close -> click
            if abs(x - thumb_x) < 40 and abs(y - thumb_y) < 40:
                pyautogui.click()

    cv2.imshow("Hand Mouse Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()
