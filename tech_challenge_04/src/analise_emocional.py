import cv2
import fer


def detectar_emocoes(frame, detector):
  """Detecta as emoções em um frame utilizando o FER Plus.

  Args:
      frame (numpy.ndarray): Frame a ser processado.
      detector (fer.FER): Detector de emoções.

  Returns:
      dict: Dicionário contendo as probabilidades de cada emoção.
  """

  # Converter o frame para RGB (se necessário)
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # Detectar rostos (você pode usar a função de detecção de rostos do OpenCV)
  # ...

  # Detectar emoções no primeiro rosto detectado (ajuste para múltiplos rostos se necessário)
  face_image = gray[y:y + h, x:x + w]
  emotions = detector.detect_emotions(face_image)

  # Retornar as emoções e suas probabilidades
  return emotions[0]['emotions']


# Exemplo de uso:
if __name__ == "__main__":
  # Carregar o detector FER Plus
  detector = fer.FER(mtl='FER2013')

  # Assumindo que você já tem um frame e a região do rosto
  frame = ...  # Substitua por seu frame
  x, y, w, h = ...  # Coordenadas do rosto

  emocoes = detectar_emocoes(frame, detector)
  print(emocoes)
