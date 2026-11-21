


import cv2
import time

cap = cv2.VideoCapture(0, cv2.CAP_MSMF)

if not cap.isOpened():
    print("Camera failed to open")
    exit()

print("SignBridge camera started")

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        print("Frame reading failed. Retrying...")
        time.sleep(0.1)
        continue

    frame = cv2.flip(frame, 1)

    cv2.imshow("SignBridge Live Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



