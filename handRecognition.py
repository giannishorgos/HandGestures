import cv2
import time
import mediapipe as mp


class DetectHand:
    def __init__(self, mode=False, number_of_hands=2, model_c=1, min_detection_confidence=0.5, max_tracking_confidence=0.5):
        self.mode = mode
        self.number_of_hands = number_of_hands
        self.min_detection_confidence = min_detection_confidence
        self.max_tracking_confidence = max_tracking_confidence
        self.model_c = model_c
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            self.mode, self.number_of_hands, self.model_c, self.min_detection_confidence, self.max_tracking_confidence)
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def drawLandmarks(self, results, image):
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                middle_x = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image.shape[1]
                middle_y = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image.shape[0]
                image = cv2.circle(
                    image, (int(middle_x), int(middle_y)), radius=10, color=(255, 255, 0), thickness=2)
                self.mp_draw.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())


def main():
    cam = cv2.VideoCapture(0)  # 0 for the first webcam on computer

    current_time = 0
    prev_time = 0

    hand_gestures = DetectHand()

    while cam.isOpened():
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        _, frame = cam.read()
        frame = cv2.putText(frame, f'fps: {str(int(fps))}', (10, 50),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hand_gestures.hands.process(frame)
        hand_gestures.drawLandmarks(results, frame)
        cv2.imshow("Webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.destroyAllWindows()
    cam.release()


if __name__ == '__main__':
    main()
