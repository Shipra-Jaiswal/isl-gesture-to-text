import numpy as np
import tensorflow as tf
import mediapipe as mp
import cv2

print("NumPy:", np.__version__)
print("TensorFlow:", tf.__version__)
print("MediaPipe:", mp.__version__)

print("MediaPipe Solutions Exists:")
print(hasattr(mp, "solutions"))
