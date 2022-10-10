import mediapipe as mp


class DetectHand:
    def __init__(self, mode=False, number_of_hands=2, min_detection_confidence=0.5, max_tracking_confidence=0.5):
        self.mode = mode
        self.number_of_hands = number_of_hands
        self.min_detection_confidence = min_detection_confidence
        self.max_tracking_confidence = max_tracking_confidence
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            self.mode, self.number_of_hands, self.min_detection_confidence, self.max_tracking_confidence)
        self.mp_draw = mp.solutions.drawing_utils
