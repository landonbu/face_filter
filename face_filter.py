import numpy as np
import cv2

cap = cv2.VideoCapture(0)
kernel = np.ones((21, 21), 'uint8')
face_cascade = cv2.CascadeClassifier(
    'src\Face Filter Project\haarcascade_frontalface_alt.xml')

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    faces = face_cascade.detectMultiScale(
        frame, scaleFactor=1.2, minSize=(20, 20))

    for (x, y, w, h) in faces:
        frame[y:y+h, x:x+w, :] = cv2.dilate(frame[y:y+h, x:x+w, :], kernel)

        # Display the resulting frame
        cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
