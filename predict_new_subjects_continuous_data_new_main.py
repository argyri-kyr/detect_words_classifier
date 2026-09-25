# predict_new_subjects_continuous_data_new_main.py
#
# Independent alternative to predict_new_subjects_main.py.
#
# Same subject-folder input structure, but instead of:
#   WhisperX transcription -> timestamps -> slice_audio -> predict_one_audio x24
# this pipeline does:
#   resolve + clean raw audio -> predict_continuous_audio (single scan, no slicing)
#
# Why: WhisperX/forced-alignment can misalign downstream words when a
# child says off-script speech ("I don't know this word", "well done").
# This pipeline never depends on alignment succeeding.
#
# Usage:
#   python predict_new_subjects_continuous_data_new_main.py
#
# Input:
#   folder1/folder2/new_subjects/subject_{id}/
#
# Output:
#   folder1/folder2/new_subjects/subject_{id}/scan_predictions_{id}.csv

import os
import sys
import inspect
import pandas as pd


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
detect_words_path = os.path.join(parentdir, "detect_words")

# ----------------------------------------------------------------
# Step 1 — Import only audio resolve/clean from detect_words
# (no WhisperX, no feature_pipeline, no slicing — this pipeline
# never depends on transcription or alignment succeeding)
# ----------------------------------------------------------------
sys.path.insert(0, detect_words_path)

from edit_audio import resolve_audio_path
from clean_audio_pipeline import clean_audio_pipeline

# ----------------------------------------------------------------
# Step 2 — Remove detect_words from path AND module cache
# so classifier imports use the classifier's globalVars.py
# ----------------------------------------------------------------
sys.path.remove(detect_words_path)

modules_to_remove = [
    name for name, mod in sys.modules.items()
    if hasattr(mod, "__file__")
    and mod.__file__ is not None
    and detect_words_path in mod.__file__
]
for name in modules_to_remove:
    del sys.modules[name]

# ----------------------------------------------------------------
# Step 3 — Now import from classifier project safely
# ----------------------------------------------------------------
sys.path.insert(0, currentdir)

import globalVars as gv
from globalVars import WAV2VEC2_MODEL_NAME, EMBEDDING_LAYER_PER_WORD
from continuous_scan import predict_continuous_audio
from wav2vec2_features import load_wav2vec2


# ----------------------------------------------------------------
# Settings
# ----------------------------------------------------------------

# Subject IDs to process — same style as predict_new_subjects_main.py
SUBJECT_IDS = [1976]   # <- add your new subject IDs here


def load_words_in_order():
    """
    Loads the 24 target words in the fixed order they're expected to
    be read, from the same source text file used elsewhere in the
    project (same pattern as predict_new_subjects_main.py's
    expected_words loading).

    This order matters — the scanner's cursor moves forward through
    it and never looks backward.
    """

    text_path = os.path.join(gv.folder1, f"texts/{gv.text}_1920/alltext.txt")
    with open(text_path, encoding="utf-8") as f:
        words = f.read().split()

    return words[:24]


def get_model_path(word):
    """
    Returns the path to model.joblib for a given word,
    using the correct embedding layer from EMBEDDING_LAYER_PER_WORD.
    """

    safe_model_name = WAV2VEC2_MODEL_NAME.replace("/", "__")
    layer = EMBEDDING_LAYER_PER_WORD.get(word, gv.EMBEDDING_LAYER)
    safe_word = word.replace(" ", "_")

    return os.path.join(
        "outputs",
        safe_model_name,
        f"layer_{layer}",
        safe_word,
        "model.joblib"
    )


def get_metadata_csv(word):
    """
    Returns the path to the training metadata.csv for a given word,
    used by continuous_scan.py to estimate expected clip duration
    (for sizing the scan window).
    """

    safe_word = word.replace(" ", "_")
    return os.path.join("data", safe_word, "metadata.csv")


def get_features_csv(word):
    """
    Returns the path to the training features.csv for a given word,
    used by continuous_scan.py to compute a neutral phonetic_score
    fallback value (median phonetic_score among correct samples).
    """

    safe_model_name = WAV2VEC2_MODEL_NAME.replace("/", "__")
    layer = EMBEDDING_LAYER_PER_WORD.get(word, gv.EMBEDDING_LAYER)
    safe_word = word.replace(" ", "_")

    return os.path.join(
        "outputs",
        safe_model_name,
        f"layer_{layer}",
        safe_word,
        "features.csv"
    )


def predict_subject_continuous(sub_id, sub_path, words_in_order, wav2vec2_processor, wav2vec2_model_loaded):
    """
    Run the continuous-scan pipeline for one subject:
        1. Resolve raw audio (handles .avi -> .wav conversion)
        2. Noise removal
        3. predict_continuous_audio() — single full-recording scan,
           no transcription, no timestamps, no slicing

    wav2vec2_processor / wav2vec2_model_loaded are pre-loaded once in
    main() and reused across all subjects, avoiding a reload per subject.

    Returns:
        DataFrame with one row per word and final_decision column.
        None if the subject cannot be processed.
    """

    print(f"\n{'=' * 60}")
    print(f"  Subject {sub_id}")
    print(f"{'=' * 60}")

    # ----------------------------------------------------------------
    # Step 1A: Resolve audio path (handles .avi → .wav conversion)
    # ----------------------------------------------------------------
    raw_audio_path = resolve_audio_path(sub_id, sub_path, gv.text)
    if raw_audio_path is None:
        print(f"  [{sub_id}] No audio found — skipping.")
        return None

    # ----------------------------------------------------------------
    # Step 1B: Noise removal
    # ----------------------------------------------------------------
    clean_audio_path = clean_audio_pipeline(sub_id, raw_audio_path)

    # ----------------------------------------------------------------
    # Step 2: Scan the full recording directly — no slicing, no
    # WhisperX timestamps required.
    # ----------------------------------------------------------------
    print(f"  [{sub_id}] Scanning full recording ({len(words_in_order)} target words)...")

    try:
        results = predict_continuous_audio(
            audio_path=clean_audio_path,
            words_in_order=words_in_order,
            get_model_path_fn=get_model_path,
            get_metadata_csv_fn=get_metadata_csv,
            get_features_csv_fn=get_features_csv,
            subject_id=str(sub_id),
            processor=wav2vec2_processor,
            wav2vec2_model=wav2vec2_model_loaded,
        )
    except Exception as e:
        print(f"  [{sub_id}] ERROR during continuous scan: {e}")
        return None

    for row in results:
        prob = row.get("probability_correct")
        prob_str = f"{prob:.3f}" if prob is not None else "N/A"
        print(f"  [{sub_id}] {row['target_word']:<25} → {row['final_decision']:<20} ({prob_str})")

    results_df = pd.DataFrame(results)

    # Save per-subject predictions CSV
    output_path = os.path.join(sub_path, f"scan_predictions_{sub_id}.csv")
    results_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n  [{sub_id}] Saved: {output_path}")

    return results_df


# AFTER
def main():
    new_subjects_base = os.path.join(
        gv.folder1,
        gv.folder2,
        "new_subjects",
    )

    words_in_order = load_words_in_order()

    print("Loading wav2vec2 model...")
    wav2vec2_processor, wav2vec2_model_loaded = load_wav2vec2(WAV2VEC2_MODEL_NAME)

    all_results = []

    for sub_id in SUBJECT_IDS:
        sub_path = os.path.join(new_subjects_base, f"subject_{sub_id}")

        if not os.path.exists(sub_path):
            print(f"Subject folder not found: {sub_path} — skipping.")
            continue

        result_df = predict_subject_continuous(
            sub_id,
            sub_path,
            words_in_order,
            wav2vec2_processor,
            wav2vec2_model_loaded,
        )

        if result_df is not None:
            all_results.append(result_df)

    # ----------------------------------------------------------------
    # Save combined results across all subjects
    # ----------------------------------------------------------------
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)

        combined_path = os.path.join(new_subjects_base, "all_scan_predictions.csv")
        combined_df.to_csv(combined_path, index=False, encoding="utf-8-sig")

        print(f"\n{'=' * 60}")
        print("  ALL SUBJECTS FINISHED (continuous scan)")
        print(f"{'=' * 60}\n")
        print(f"Saved combined results: {combined_path}")
        print()

        print(f"{'Subject':<12} {'Word':<25} {'Decision':<20} {'Prob':>6}")
        print("-" * 68)
        for _, row in combined_df.iterrows():
            prob = f"{row['probability_correct']:.3f}" if pd.notna(row['probability_correct']) else "N/A"
            print(f"{str(row['subject_id']):<12} {row['target_word']:<25} {row['final_decision']:<20} {prob:>6}")


if __name__ == "__main__":
    main()