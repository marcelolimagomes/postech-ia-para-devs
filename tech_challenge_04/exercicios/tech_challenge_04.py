# %%
# Configurando python environment. Necessário ter instalado uma GPU Nvdia (no meu caso RTX-3060) + Anaconda
# conda create -n tc5 -c rapidsai -c conda-forge -c nvidia rapids=24.2 python=3.10 'cuda-version>=12.0,<=12.5' tensorflow[and-cuda]==2.15.0 'pytorch=*=*cuda*' torchvision deepface ultralytics
# conda create -n tc4 -c rapidsai -c conda-forge -c nvidia rapids=24.10 python=3.11 'cuda-version>=12.0,<=12.5' 'pytorch=*=*cuda*' ultralytics
# pip install opencv-python cvzone tqdm mediapipe

# >>>Baixa classificador
#!wget -O classifier0.tflite -q https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite0/float32/1/efficientnet_lite0.tflite
#!wget -O classifier2.tflite -q https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite2/float32/1/efficientnet_lite2.tflite

# %%
import warnings
warnings.filterwarnings('ignore')

# %%
import torchvision
import torch
print(torch.__version__)
print('CUDA available: ' + str(torch.cuda.is_available()))

# %%
import tensorflow as tf
print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))

# %%
import cv2
import cvzone
from deepface import DeepFace
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.components import processors
from mediapipe.tasks.python import vision
from tqdm import tqdm

# %%
SMALL = False
if SMALL:
  video_path = './data/resized_video.mp4'
  output_video_path = './data/resized_video_result.mp4'
else:
  # video_path = './data/video.mp4'
  # output_video_path = './data/video_result.mp4'
  # video_path = './data/trailer_senna.mp4'
  # output_video_path = './data/trailer_senna_result.mp4'
  video_path = './data/compilado_esporte_resized.mp4'
  output_video_path = './data/compilado_esporte_result.mp4'


def get_cap(video_path):
  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
    print("Cannot open video")

  return cap


cap = get_cap(video_path)

# Create an ImageClassifier object.
VisionRunningMode = mp.tasks.vision.RunningMode
base_options = python.BaseOptions(model_asset_path='classifier2.tflite')
options = vision.ImageClassifierOptions(
    base_options=base_options,
    max_results=4,
    running_mode=VisionRunningMode.VIDEO,
    display_names_locale='pt')
classifier = vision.ImageClassifier.create_from_options(options)

# %%
frames_scape = 1
count = 1

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f'Original: width: {width} - height: {height} - fps: {fps} - total_frames: {total_frames}')

middle = int(width / 4)
bottom = height - 10

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
text = 'Action: Undefined'
# Loop para processar cada frame do vídeo com barra de progresso
for frame_index in tqdm(range(total_frames), desc="Processando vídeo"):
  ret, frame = cap.read()
  if count < frames_scape:
    count += 1
    continue

  count = 1

  # if frame is read correctly ret is True
  if not ret:
    print("Can't receive frame (stream end?). Exiting ...")
    break

  # Detectar rostos
  rostos_detectados = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, detector_backend='yolov8')
  # Convert the frame received from OpenCV to a MediaPipe’s Image object.
  mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
  # Calculate the timestamp of the current frame
  frame_timestamp_ms = int(1000 * frame_index / fps)
  # Perform image classification on the video frame.
  classification_result = classifier.classify_for_video(mp_image, frame_timestamp_ms)

  for face in rostos_detectados:
    # print(face)
    x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
    cvzone.cornerRect(frame, [x, y, w, h], l=9, t=2, rt=1)
    dominant_emotion = face['dominant_emotion']
    # cvzone.putTextRect(frame, dominant_emotion, [x, y - 10],  scale = 1, font = cv2.FONT_HERSHEY_SIMPLEX, colorB = None)
    cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

  # Process the classification result. In this case, visualize it.
  top_category = classification_result.classifications[0].categories[0]
  if top_category.score >= .20:
    text = f"Action: {top_category.category_name} ({top_category.score:.2f})"
  cv2.putText(frame, text, (middle, bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

  # cv2.imshow('Detecção de Padrões', frame)
  out.write(frame)

  k = cv2.waitKey(30) & 0xff
  if k == 27:  # Esc Key
    break

# Close the window
cap.release()
out.release()
# De-allocate any associated memory usage
cv2.destroyAllWindows()
