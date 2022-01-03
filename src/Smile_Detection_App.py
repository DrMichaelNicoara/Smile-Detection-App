import cv2

face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
smile_data = cv2.CascadeClassifier("haarcascade_smile.xml")
webcam = cv2.VideoCapture(0)

while True:
    success, frame = webcam.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coordinates = face_data.detectMultiScale(gray_frame)

    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        face_crop = gray_frame[y:y+h, x:x+w]
        smile_coordinates = smile_data.detectMultiScale(face_crop, scaleFactor = 1.17, minNeighbors = 20)

        if len(smile_coordinates) > 0:
            cv2.putText(frame, "smiling", (x, y+h+40), fontScale = 3, fontFace = cv2.FONT_HERSHEY_PLAIN, color = (255, 255, 255))

    cv2.imshow("Face", frame)

    # Hit q or Q to exit program
    key = cv2.waitKey(1);
    if key == 81 or key == 113:
        break

webcam.release()