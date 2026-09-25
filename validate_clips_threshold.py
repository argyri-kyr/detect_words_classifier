# validate_clips_threshold.py
# Run this script once to check if similarity >= 60 correctly
# identifies bad clips for subject 1973.
# Output: a CSV showing per-clip similarity, Whisper transcription,
# and pass/fail — so you can manually check if the threshold is right.

import os
import sys
import inspect
import pandas as pd

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
detect_words_path = os.path.join(parentdir, "detect_words")

sys.path.insert(0, detect_words_path)
from clean_audio_pipeline import clean_audio_pipeline
sys.path.remove(detect_words_path)
modules_to_remove = [
    name for name, mod in sys.modules.items()
    if hasattr(mod, "__file__") and mod.__file__ is not None
    and detect_words_path in mod.__file__
]
for name in modules_to_remove:
    del sys.modules[name]

sys.path.insert(0, currentdir)
import globalVars as gv
from whisper_features import load_whisper_model
from load_phonetic_scores import normalize_greek_phonetic
from rapidfuzz import fuzz


def get_whisper_similarity_and_text(audio_path, target_word, whisper_model):
    """
    Transcribe clip freely with Whisper and return both
    the similarity score and the raw transcription text.
    """
    result = whisper_model.transcribe(
        audio_path,
        language="el",
        without_timestamps=True,
        fp16=False,
    )
    whisper_text = result["text"].strip()

    if not whisper_text:
        return 0.0, ""

    whisper_norm = normalize_greek_phonetic(whisper_text)
    target_norm = normalize_greek_phonetic(target_word)
    similarity = fuzz.ratio(whisper_norm, target_norm)

    return float(similarity), whisper_text


def extract_word_from_clip_filename(filename):
    basename = os.path.splitext(os.path.basename(filename))[0]
    parts = basename.split("_", 1)
    if len(parts) == 2:
        return parts[1]
    return basename


def main():
    sub_id = 1973   # ← change if needed
    threshold = 60  # ← the threshold we want to test

    clips_dir = os.path.join(
        gv.folder1,
        gv.folder2,
        "new_subjects",
        f"subject_{sub_id}",
        f"decodewordspseudo_word_clips_{sub_id}",
    )

    if not os.path.exists(clips_dir):
        print(f"Clips folder not found: {clips_dir}")
        return

    clip_files = sorted([
        f for f in os.listdir(clips_dir)
        if f.endswith(".wav")
    ])

    print(f"Loading Whisper model ({gv.WHISPER_MODEL_SIZE})...")
    whisper_model = load_whisper_model()

    print(f"\nValidating {len(clip_files)} clips for subject {sub_id}...")
    print(f"Threshold: similarity >= {threshold} = VALID\n")

    rows = []

    for clip_file in clip_files:
        word = extract_word_from_clip_filename(clip_file)
        clip_path = os.path.join(clips_dir, clip_file)

        similarity, whisper_text = get_whisper_similarity_and_text(
            clip_path, word, whisper_model
        )

        passed = similarity >= threshold

        status = "✅ VALID" if passed else "❌ INVALID"
        print(f"  {clip_file:<35} similarity={similarity:5.1f}  "
              f"whisper='{whisper_text}'  {status}")

        rows.append({
            "clip_file": clip_file,
            "expected_word": word,
            "whisper_text": whisper_text,
            "similarity": similarity,
            "passes_threshold": passed,
            "threshold": threshold,
        })

    df = pd.DataFrame(rows)

    # Summary
    valid = df["passes_threshold"].sum()
    invalid = len(df) - valid
    print(f"\nSummary: {valid}/{len(df)} valid, {invalid}/{len(df)} invalid")
    print(f"Invalid clips: {df[~df['passes_threshold']]['expected_word'].tolist()}")

    # Save results
    output_path = os.path.join(
        gv.folder1, gv.folder2, "new_subjects",
        f"subject_{sub_id}",
        f"clip_validation_{sub_id}.csv"
    )
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved validation results: {output_path}")

    # Show what threshold would be needed to correctly
    # classify the clips you know are wrong
    print(f"\n=== THRESHOLD ANALYSIS ===")
    print(f"Distribution of similarity scores:")
    print(df[["expected_word", "similarity", "passes_threshold"]]
          .sort_values("similarity")
          .to_string(index=False))


if __name__ == "__main__":
    main()