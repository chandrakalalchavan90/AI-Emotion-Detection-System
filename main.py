from deepface import DeepFace
import cv2
import time
import csv
from datetime import datetime

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

if not cap.isOpened():
    print("Camera not found")
    exit()

prev_time = 0

file = open('emotion_log.csv', 'a', newline='')

writer = csv.writer(file)

writer.writerow(["Time", "Emotion", "Confidence"])

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # FPS calculation
    curr_time = time.time()

    fps = 1 / (curr_time - prev_time)

    prev_time = curr_time

    try:

        result = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False
        )

        emotion = result[0]['dominant_emotion']

        confidence = result[0]['emotion'][emotion]

        current_time = datetime.now().strftime("%H:%M:%S")

        # Save emotion data
        writer.writerow([
            current_time,
            emotion,
            round(confidence, 2)
        ])

        # Emotion colors
        color = (255, 255, 255)

        if emotion == "happy":
            color = (0, 255, 0)

        elif emotion == "angry":
            color = (0, 0, 255)

        elif emotion == "neutral":
            color = (0, 255, 255)

        # Draw rectangle around face
        for (x, y, w, h) in faces:

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                color,
                3
            )

        # Show emotion text
        cv2.putText(
            frame,
            f'{emotion} : {confidence:.2f}%',
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

    except Exception as e:
        print("Error:", e)

    # Show FPS
    cv2.putText(
        frame,
        f'FPS: {int(fps)}',
        (50, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.imshow("AI Emotion Detection System", frame)

    key = cv2.waitKey(1)

    # Quit program
    if key & 0xFF == ord('q'):
        break

    # Save screenshot
    if key & 0xFF == ord('s'):

        filename = f"screenshots/face_{int(time.time())}.jpg"

        cv2.imwrite(filename, frame)

        print(f"Screenshot Saved: {filename}")

cap.release()

file.close()

cv2.destroyAllWindows()