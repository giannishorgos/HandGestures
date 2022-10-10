import cv2
import mediapipe as mp
import time


def main():
    cam = cv2.VideoCapture(0)  # 0 for the first webcam on computer

    current_time = 0
    prev_time = 0

    while cam.isOpened():
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        success, frame = cam.read()
        if success:
            frame = cv2.putText(frame, f'fps: {str(int(fps))}', (10, 50),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
            cv2.imshow("Webcam", frame)
        else:
            break

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cam.release()
    cam.destroyAllWindows()


if __name__ == '__main__':
    main()
