

import cv2
import time

try:
    from .test_hand import open_camera
except ImportError:
    from test_hand import open_camera


def main():
    cap, frame = open_camera()

    if cap is None:
        raise RuntimeError("Unable to start camera")

    failed_reads = 0

    try:
        print("SignBridge Live Camera Started")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        while True:

            success, new_frame = cap.read()

            if not success or new_frame is None:
                failed_reads += 1
                print("Frame reading failed:", failed_reads)

                if failed_reads >= 10:
                    print("Trying to reconnect camera...")

                    cap.release()
                    time.sleep(1)

                    cap, frame = open_camera()

                    if cap is None:
                        print("Camera reconnection failed")
                        break

                    failed_reads = 0

                continue

            failed_reads = 0

            frame = cv2.flip(new_frame, 1)

            cv2.imshow("SignBridge Live Camera", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        if cap is not None:
            cap.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
