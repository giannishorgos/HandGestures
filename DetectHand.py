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
                self.mp_draw.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())
