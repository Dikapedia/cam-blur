import argparse
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
draw_utils = mp.solutions.drawing_utils


class HandBlurCamera:
    def __init__(
        self, resolution=(640, 480), blur_kernel=(71, 71), fullscreen=False
    ):
        self.resolution = resolution
        self.blur_kernel = (max(3, blur_kernel[0] | 1), max(3, blur_kernel[1] | 1))
        self.fullscreen = fullscreen
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Tidak dapat membuka kamera.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

    def _is_peace_sign(self, lm):
        index_extended = lm[8].y < lm[6].y
        middle_extended = lm[12].y < lm[10].y
        ring_folded = lm[16].y > lm[14].y
        pinky_folded = lm[20].y > lm[18].y

        return index_extended and middle_extended and ring_folded and pinky_folded

    def run(self):
        cv2.namedWindow("Blur Camera", cv2.WINDOW_NORMAL)
        if self.fullscreen:
            cv2.setWindowProperty(
                "Blur Camera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )
        else:
            cv2.resizeWindow("Blur Camera", self.resolution[0] * 2, self.resolution[1] * 2)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Mirror horizontally
            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            peace_detected = False
            if results.multi_hand_landmarks:
                for hand_lm in results.multi_hand_landmarks:
                    if self._is_peace_sign(hand_lm.landmark):
                        peace_detected = True

            if peace_detected:
                frame = cv2.GaussianBlur(
                    frame, self.blur_kernel, 0
                )

            cv2.imshow("Blur Camera", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Blur seluruh kamera saat gesture peace sign."
    )
    parser.add_argument("--resolution", type=int, nargs=2, default=(640, 480))
    parser.add_argument("--blur-kernel", type=int, nargs=2, default=(71, 71))
    parser.add_argument("--fullscreen", action="store_true", help="Mode fullscreen")
    parser.add_argument("--flip", type=int, default=1, help="Flip 1=horiz mirror, 0=none, -1=both")
    args = parser.parse_args()

    cam = HandBlurCamera(
        resolution=args.resolution, blur_kernel=args.blur_kernel, fullscreen=args.fullscreen
    )
    cam.run()
