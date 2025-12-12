import cv2
import mediapipe as mp
import pyautogui
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

last_action_time = 0
cooldown = 1.5  # jeda

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                wrist = hand_landmarks.landmark[0]
                x_pix = int(wrist.x * w)
                y_pix = int(wrist.y * h)

                cv2.circle(frame, (x_pix, y_pix), 10, (0, 255, 0), -1)

                # batas zona
                left_zone = int(0.3 * w)
                right_zone = int(0.7 * w)
                cv2.line(frame, (left_zone, 0), (left_zone, h), (255, 0, 0), 2)
                cv2.line(frame, (right_zone, 0), (right_zone, h), (0, 0, 255), 2)

                now = time.time()

                # zona kiri -> previous slide
                if x_pix < left_zone and now - last_action_time > cooldown:
                    print("PREV SLIDE")
                    pyautogui.press("left")
                    last_action_time = now

                # zona kanan -> next slide
                elif x_pix > right_zone and now - last_action_time > cooldown:
                    print("NEXT SLIDE")
                    pyautogui.press("right")
                    last_action_time = now

        cv2.imshow("Hand Slide Control", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
