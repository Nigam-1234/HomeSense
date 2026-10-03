"""
features.py — convert raw audio clips into log-mel spectrogram features
used to train and run the HomeSense classifier.

Usage:
    python src/features.py
"""
import os
import glob
import numpy as np
import librosa
from tqdm import tqdm

from config import (
    CLASSES,
    SAMPLE_RATE,
    CLIP_DURATION_SEC,
    N_MELS,
    HOP_LENGTH,
    N_FFT,
    SAMPLES_DIR,
    FEATURES_DIR,
)


def load_and_window(path, sr=SAMPLE_RATE, duration=CLIP_DURATION_SEC):
    """Load an audio file, resample, and pad/trim to a fixed duration."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y


def extract_logmel(y, sr=SAMPLE_RATE):
    """Compute a log-mel spectrogram feature array for one audio window."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    # Normalize to roughly [0, 1] for stable training.
    log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-6)
    return log_mel.astype(np.float32)


def build_dataset():
    os.makedirs(FEATURES_DIR, exist_ok=True)
    X, y_labels = [], []

    for class_idx, cls in enumerate(CLASSES):
        folder = os.path.join(SAMPLES_DIR, cls)
        files = glob.glob(os.path.join(folder, "*.wav"))
        if not files:
            print(f"[warn] No .wav files found for class '{cls}' in {folder}")
            continue

        for path in tqdm(files, desc=f"Processing {cls}"):
            audio = load_and_window(path)
            feat = extract_logmel(audio)
            X.append(feat)
            y_labels.append(class_idx)

    if not X:
        print("No audio files found anywhere under data/samples/. "
              "Run src/data_prep.py and add some .wav clips first.")
        return

    X = np.stack(X)[..., np.newaxis]  # add channel dim for CNN input
    y_labels = np.array(y_labels)

    np.save(os.path.join(FEATURES_DIR, "X.npy"), X)
    np.save(os.path.join(FEATURES_DIR, "y.npy"), y_labels)
    print(f"\nSaved {X.shape[0]} samples -> {FEATURES_DIR}/X.npy, y.npy")
    print(f"Feature shape per sample: {X.shape[1:]}")


if __name__ == "__main__":
    build_dataset()
