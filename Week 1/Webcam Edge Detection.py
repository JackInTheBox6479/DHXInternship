import cv2
import numpy as np

webcam = cv2.VideoCapture(0)

while True:
    ret, frame = webcam.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if ret:
        image = cv2.Laplacian(frame, ksize=5,  ddepth=cv2.CV_64F)
        image = np.abs(image)
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        image = np.uint8(image)

        cv2.imshow("", image)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

webcam.release()
cv2.destroyAllWindows()
