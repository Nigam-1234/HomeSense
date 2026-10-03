"""
Configuration shared across the HomeSense pipeline.
"""
import os

# Event classes the model recognizes (v1 scope — kept deliberately small
# for a 1-2 week hackathon build: depth over breadth).
CLASSES = [
    "baby_cry",
    "glass_break",
    "smoke_alarm",
    "water_running",
    "background",
]

SAMPLE_RATE = 16000          # Hz — good tradeoff of quality vs on-device compute
CLIP_DURATION_SEC = 2.0      # length of each audio window fed to the model
N_MELS = 64                  # mel bands for the spectrogram features
HOP_LENGTH = 256
N_FFT = 1024

RAW_DATA_DIR = os.path.join("data", "raw")
SAMPLES_DIR = os.path.join("data", "samples")
FEATURES_DIR = os.path.join("data", "features")
MODELS_DIR = "models"

KERAS_MODEL_PATH = os.path.join(MODELS_DIR, "homesense_cnn.h5")
TFLITE_MODEL_PATH = os.path.join(MODELS_DIR, "homesense_cnn.tflite")

# Confidence threshold above which an event triggers an action downstream.
EVENT_CONFIDENCE_THRESHOLD = 0.75
