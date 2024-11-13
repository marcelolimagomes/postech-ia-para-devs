import cv2


def detectar_rostos(frame):
  """Detecta rostos em um frame utilizando o classificador Haar Cascade.

  Args:
      frame (numpy.ndarray): Frame a ser processado.

  Returns:
      list: Lista de tuplas (x, y, w, h), representando as coordenadas e dimensões dos rostos detectados.
  """

  # Carregar o classificador Haar Cascade
  face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

  # Converter o frame para tons de cinza
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # Detectar rostos
  rostos = face_cascade.detectMultiScale(gray, 1.1, 4)

  return rostos


def desenhar_retangulos(frame, rostos):
  """Desenha retângulos ao redor dos rostos detectados.

  Args:
      frame (numpy.ndarray): Frame a ser modificado.
      rostos (list): Lista de tuplas (x, y, w, h) representando os rostos.
  """

  for (x, y, w, h) in rostos:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)


# Exemplo de uso:
if __name__ == "__main__":
  # Assumindo que você já tem um frame do vídeo
  frame = ...  # Substitua por seu frame

  rostos_detectados = detectar_rostos(frame)
  desenhar_retangulos(frame, rostos_detectados)

  # Mostrar o frame com os rostos marcados
  cv2.imshow('Frame com rostos', frame)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
