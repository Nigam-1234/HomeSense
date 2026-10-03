# HomeSense

**Your phone becomes the smart-home brain — no new sensors needed.**

HomeSense runs real-time, fully offline audio event detection on-device
(designed for the iQOO 15 / Snapdragon NPU) to recognize safety- and
convenience-critical events in the home — a baby crying, glass breaking,
a smoke/gas alarm, a running tap/leak — and trigger local actions, with
zero network calls and zero cloud dependency.

## Why

Smart home safety today usually means buying a sensor for every room
(smoke detectors with WiFi, glass-break sensors, baby monitors...).
HomeSense turns the phone people already own into an ambient sensor
network using only the microphone and on-device ML.

## Status: Phase 1 prototype (audio event detection)

This prototype demonstrates the core ML pipeline: classifying short
audio clips into home-safety event categories, running fully offline.
Android integration (mic streaming + NPU-accelerated on-device inference)
is the next phase.

### Event classes (v1)
- `baby_cry`
- `glass_break`
- `smoke_alarm`
- `water_running` (tap/leak)
- `background` (everything else / silence)

## Project structure

```
homesense/
├── src/
│   ├── data_prep.py       # Download/organize public datasets (ESC-50, UrbanSound8K subset)
│   ├── features.py        # Audio -> log-mel spectrogram feature extraction
│   ├── train.py           # Train a small CNN classifier
│   ├── convert_tflite.py  # Export trained model to TFLite (NPU-ready)
│   └── infer.py           # Run inference on a live mic stream or a .wav file
├── data/samples/          # A few example clips for each event class (for demo)
├── models/                # Trained .h5 / exported .tflite models
├── notebooks/             # Exploration notebooks (optional)
└── tests/
```

## Quickstart

```bash
pip install -r requirements.txt

# 1. Prepare data (downloads a small public subset + uses your recorded samples)
python src/data_prep.py

# 2. Extract features
python src/features.py

# 3. Train the classifier
python src/train.py

# 4. Export to TFLite for on-device / NPU inference
python src/convert_tflite.py

# 5. Try live inference on a sample clip
python src/infer.py --file data/samples/glass_break/example1.wav
```

## Datasets used

- [ESC-50](https://github.com/karoldvl/ESC-50) — environmental sound classification (glass break, alarms, etc.)
- [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) — supplementary urban/home sounds
- A small set of self-recorded clips per class for demo robustness (not included in repo — see `data/samples/README.md`)

All third-party datasets are used under their respective licenses for
non-commercial hackathon/research purposes; see each dataset's page for terms.

## Roadmap

- [x] Audio event classifier (offline, Python prototype)
- [ ] TFLite export tuned for Snapdragon NPU (NNAPI delegate)
- [ ] Android app: live mic streaming + on-device inference
- [ ] Action triggers: push notification, Bluetooth smart plug/light flash
- [ ] On-stage demo script (3-4 staged events, sub-second reaction)

## Team

Built for the iQOO Hyderabad hackathon grand finale — Smart Living track.

## License

MIT (for original code in this repo). Third-party datasets retain their own licenses.
