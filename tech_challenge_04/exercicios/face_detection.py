import numpy as np
import cv2
import time


def capture_video():
  # Carregando o classificador de Haar Cascades para detecção de faces
  face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

  video_path = './video.mp4'
  cap = cv2.VideoCapture(0)
  if not cap.isOpened():
    print("Cannot open camera")
    exit()

  # while cap.isOpened():
  while True:
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
      print("Can't receive frame (stream end?). Exiting ...")
      break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectando faces no frame em escala de cinza
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    print(faces)
    # Enquanto o loop estiver em execução, o rosto será desenhado
    # Desenhando um retângulo em torno de cada face detectada
    for (x, y, w, h) in faces:
      cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('Detecção de Rosto', frame)
    if cv2.waitKey(1) == ord('q'):
      break
    time.sleep(.5)

  cap.release()
  cv2.destroyAllWindows()


if __name__ == "__main__":
  capture_video()
