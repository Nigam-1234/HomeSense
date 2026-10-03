"""Basic smoke tests for the HomeSense feature pipeline (no trained model required)."""
import numpy as np
from features import extract_logmel
from config import N_MELS, SAMPLE_RATE, CLIP_DURATION_SEC


def test_extract_logmel_shape():
    y = np.random.randn(int(SAMPLE_RATE * CLIP_DURATION_SEC)).astype(np.float32)
    feat = extract_logmel(y)
    assert feat.shape[0] == N_MELS
    assert feat.dtype == np.float32


def test_extract_logmel_range():
    y = np.random.randn(int(SAMPLE_RATE * CLIP_DURATION_SEC)).astype(np.float32)
    feat = extract_logmel(y)
    assert feat.min() >= 0.0 - 1e-5
    assert feat.max() <= 1.0 + 1e-5
