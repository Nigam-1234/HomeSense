"""
data_prep.py — organize audio samples into the data/samples/<class>/ layout
the rest of the pipeline expects.

This script does NOT auto-download ESC-50/UrbanSound8K (large, and license
terms vary) — instead it:
  1. Creates the expected folder structure.
  2. Lets you drop your own recorded .wav clips into each class folder.
  3. Prints clear instructions for pulling matching clips from ESC-50 if
     you want a bigger training set.

Usage:
    python src/data_prep.py
"""
import os
from config import CLASSES, SAMPLES_DIR


ESC50_HINTS = {
    "glass_break": ["glass_breaking"],
    "smoke_alarm": ["siren", "fire_alarm"],  # ESC-50 doesn't have a dedicated
                                              # smoke-alarm class; siren/fire
                                              # engine clips are a reasonable
                                              # stand-in or record your own.
    "baby_cry": ["crying_baby"],
    "water_running": ["water_drops", "pouring_water"],
    "background": ["breathing", "clock_tick", "footsteps"],
}


def ensure_dirs():
    for cls in CLASSES:
        path = os.path.join(SAMPLES_DIR, cls)
        os.makedirs(path, exist_ok=True)
        readme = os.path.join(path, "README.md")
        if not os.path.exists(readme):
            with open(readme, "w") as f:
                f.write(f"# {cls}\n\n")
                f.write("Drop 2-second (or longer, will be auto-windowed) "
                         f".wav clips of '{cls}' here.\n\n")
                hints = ESC50_HINTS.get(cls)
                if hints:
                    f.write(
                        "Suggested matching ESC-50 category name(s): "
                        + ", ".join(hints) + "\n"
                    )
        print(f"Ready: {path}")


def main():
    ensure_dirs()
    print("\nFolder structure created under data/samples/.")
    print("Next steps:")
    print("  1. Record a handful of clips per class on your phone (5-10 each")
    print("     is enough for a demo-quality model) and drop the .wav files")
    print("     into the matching data/samples/<class>/ folder.")
    print("  2. Optionally supplement with public clips from ESC-50:")
    print("     https://github.com/karoldvl/ESC-50 (see each folder's README")
    print("     for the suggested ESC-50 category name to pull from).")
    print("  3. Run: python src/features.py")


if __name__ == "__main__":
    main()
