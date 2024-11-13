import cv2


def carregar_video(caminho_video):
  """Carrega um vídeo a partir do caminho especificado.

  Args:
      caminho_video (str): Caminho completo do arquivo de vídeo.

  Returns:
      cv2.VideoCapture: Objeto que representa o vídeo capturado.
  """

  cap = cv2.VideoCapture(caminho_video)
  if not cap.isOpened():
    raise IOError("Não foi possível abrir o vídeo")
  return cap


def extrair_frames(cap, salvar_frames=False, caminho_saida='frames'):
  """Extrai frames de um vídeo capturado.

  Args:
      cap (cv2.VideoCapture): Objeto que representa o vídeo capturado.
      salvar_frames (bool, opcional): Indica se os frames devem ser salvos em disco.
      caminho_saida (str, opcional): Caminho para salvar os frames.

  Yields:
      numpy.ndarray: Próximo frame do vídeo.
  """

  i = 0
  while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
      break
    if salvar_frames:
      cv2.imwrite(f'{caminho_saida}/frame_{i}.jpg', frame)
    i += 1
    yield frame


def redimensionar_frame(frame, nova_largura, nova_altura):
  """Redimensiona um frame para as dimensões especificadas.

  Args:
      frame (numpy.ndarray): Frame a ser redimensionado.
      nova_largura (int): Nova largura do frame.
      nova_altura (int): Nova altura do frame.

  Returns:
      numpy.ndarray: Frame redimensionado.
  """

  return cv2.resize(frame, (nova_largura, nova_altura))


# Exemplo de uso:
if __name__ == "__main__":
  caminho_video = 'meu_video.mp4'
  cap = carregar_video(caminho_video)

  for frame in extrair_frames(cap, salvar_frames=True):
    frame_redimensionado = redimensionar_frame(frame, 224, 224)  # Redimensionar para 224x224
    # Aplicar outras transformações, se necessário
