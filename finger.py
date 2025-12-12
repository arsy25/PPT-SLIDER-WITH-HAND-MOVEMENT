import cv2
import mediapipe as mp
import pyautogui
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

last_action_time = 0
cooldown = 0.8  # detik

def count_fingers(hand_landmarks, w, h):
    # indeks tip dan pip untuk 4 jari (tidak termasuk jempol)
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]

    fingers = 0

    for tip_id, pip_id in zip(tips, pips):
        tip = hand_landmarks.landmark[tip_id]
        pip = hand_landmarks.landmark[pip_id]
        if tip.y < pip.y:  # tip di atas pip -> jari terangkat
            fingers += 1

    return fingers

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

                fingers = count_fingers(hand_landmarks, w, h)

                cv2.putText(frame, f"Fingers: {fingers}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                now = time.time()

                if fingers == 1 and now - last_action_time > cooldown:
                    print("PREV SLIDE (1 finger)")
                    pyautogui.press("left")
                    last_action_time = now

                elif fingers == 2 and now - last_action_time > cooldown:
                    print("NEXT SLIDE (2 fingers)")
                    pyautogui.press("right")
                    last_action_time = now

        cv2.imshow("Finger Slide Control", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
