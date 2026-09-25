# predict_new_subjects_main.py
#
# Full prediction pipeline for new subjects with continuous audio recordings.
#
# What it does:
#   For each new subject, it runs two major stages:
#
#   Stage A — Audio processing (uses detect_words project):
#       1. Resolve audio file path (handles .avi -> .wav conversion)
#       2. Clean audio (noise removal via clean_audio_pipeline)
#       3. Run WhisperX forced alignment to get word timestamps -> Excel file
#       4. Slice the continuous audio into one .wav clip per word
#
#   Stage B — Classification (uses this classifier project):
#       5. For each word clip, load the correct model.joblib
#       6. Call predict_one_audio() -> final_decision per word
#       7. Save per-subject predictions CSV
#
#   Final output:
#       - subject_{id}/predictions_{id}.csv  (one row per word)
#       - new_subjects/all_predictions.csv   (all subjects combined)
#
# Important: detect_words and this classifier both have a globalVars.py.
# The sys.path manipulation below ensures only the classifier's globalVars
# is imported after the detect_words functions are loaded.
#
# Usage:
#   python predict_new_subjects_main.py
#   (edit subject_ids list inside main() to add new subjects)

import os
import sys
import inspect
import pandas as pd


currentdir    = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir     = os.path.dirname(currentdir)
detect_words_path = os.path.join(parentdir, "detect_words")

# ----------------------------------------------------------------
# Step 1 — Import from detect_words using its own path.
# detect_words has its own globalVars.py (different from ours).
# We import what we need before removing it from sys.path.
# ----------------------------------------------------------------
sys.path.insert(0, detect_words_path)

from edit_audio import resolve_audio_path           # handles .avi -> .wav conversion
from clean_audio_pipeline import clean_audio_pipeline  # noise removal
from load_faster_whisperx_model import load_whisperx_model  # WhisperX model loader
from feature_pipeline import feature_pipeline        # transcription + forced alignment
from pronounce_words_main import slice_audio, find_timestamps_file  # audio slicing

# ----------------------------------------------------------------
# Step 2 — Remove detect_words from sys.path AND from module cache.
# This prevents detect_words' globalVars from shadowing ours when
# the classifier modules are imported below.
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
# Step 3 — Import classifier modules with the classifier's globalVars.
# After the cleanup above, all imports use the classifier's namespace.
# ----------------------------------------------------------------
sys.path.insert(0, currentdir)

import globalVars as gv
from globalVars import (
    WAV2VEC2_MODEL_NAME,
    WAV2VEC2_MODEL_PER_WORD,
    EMBEDDING_LAYER_PER_WORD,
    EMBEDDING_LAYERS_COMBINE_PER_WORD,
)
from predict import predict_one_audio
from wav2vec2_features import load_wav2vec2


# ----------------------------------------------------------------
# Global settings
# ----------------------------------------------------------------
TEXT          = "decodewordspseudo"  # text label used in filename conventions
PADDING_START = 300   # ms of audio kept before detected word start
PADDING_END   = 0     # ms of audio kept after detected word end

# Read the 24 target words from text file at runtime.
# This ensures the word list always matches the detect_words pipeline.
with open(gv.folder1 + f'texts/{gv.text}_1920/alltext.txt', encoding="utf-8") as f:
    TARGET_WORDS = f.read().split()
TARGET_WORDS = TARGET_WORDS[:24]


# ----------------------------------------------------------------
# MODEL PATH RESOLUTION
# ----------------------------------------------------------------

def get_model_path(word):
    """
    Returns the path to model.joblib for a given word.

    Accounts for per-word model selection and per-word layer config:

    Model selection (WAV2VEC2_MODEL_PER_WORD):
        Some words use skylord/wav2vec2-large-xlsr-greek-2 instead of
        lighteternal/wav2vec2-large-xlsr-53-greek. The model name
        determines which subfolder under outputs/ to look in.

    Layer folder selection:
        - Words in EMBEDDING_LAYERS_COMBINE_PER_WORD -> "layer_combined"
        - Other words -> "layer_N" where N is from EMBEDDING_LAYER_PER_WORD

    Example output paths:
        outputs/lighteternal__wav2vec2-large-xlsr-53-greek/layer_combined/ακαδημαϊκά/model.joblib
        outputs/skylord__wav2vec2-large-xlsr-greek-2/layer_combined/μετεγγραφές/model.joblib
        outputs/lighteternal__wav2vec2-large-xlsr-53-greek/layer_12/δεσμεύω/model.joblib
    """

    # Determine which wav2vec2 model was used for this word
    model_name      = WAV2VEC2_MODEL_PER_WORD.get(word, WAV2VEC2_MODEL_NAME)
    safe_model_name = model_name.replace("/", "__")

    # Determine layer folder
    if word in EMBEDDING_LAYERS_COMBINE_PER_WORD:
        layer_folder = "layer_combined"
    else:
        layer        = EMBEDDING_LAYER_PER_WORD.get(word, gv.EMBEDDING_LAYER)
        layer_folder = f"layer_{layer}"

    safe_word = word.replace(" ", "_")

    return os.path.join(
        "outputs",
        safe_model_name,
        layer_folder,
        safe_word,
        "model.joblib"
    )


# ----------------------------------------------------------------
# CLIP FILENAME PARSING
# ----------------------------------------------------------------

def extract_word_from_clip_filename(filename):
    """
    Extracts the target word from a word-clip filename.

    Clip filenames follow the pattern: {index}_{word}.wav
    produced by pronounce_words_main.slice_audio().

    Examples:
        0000_ποίημα.wav    -> ποίημα
        0001_λουλούδια.wav -> λουλούδια
        0023_υπεύθυνος.wav -> υπεύθυνος
    """

    basename = os.path.splitext(os.path.basename(filename))[0]
    parts    = basename.split("_", 1)  # split on first underscore only
    if len(parts) == 2:
        return parts[1]
    return basename


# ----------------------------------------------------------------
# PER-SUBJECT PIPELINE
# ----------------------------------------------------------------

def predict_subject(
    sub_id,
    sub_path,
    whisperx_device,
    whisperx_model,
    whisperx_align_model,
    whisperx_metadata,
    expected_words,
    wav2vec2_cache,
):
    """
    Runs the complete two-stage pipeline for one subject.

    Stage A — Audio processing (detect_words):
        1. Resolve audio: find the .wav or convert from .avi.
        2. Clean audio: remove background noise.
        3. Transcription: if no timestamps Excel exists yet, run
           WhisperX forced alignment to produce it.
        4. Slicing: cut the continuous audio into one clip per word
           using the timestamps. Clips saved to
           {sub_path}/{TEXT}_word_clips_{sub_id}/.

    Stage B — Classification (this project):
        5. For each clip:
               a. Determine the correct wav2vec2 model for this word
                  (from wav2vec2_cache — loaded once, reused).
               b. Find the model.joblib path via get_model_path().
               c. Call predict_one_audio() -> final_decision.
               d. Append result row.
        6. Save per-subject predictions CSV.

    wav2vec2_cache:
        dict mapping model_name -> (processor, model).
        Each wav2vec2 model is loaded at most once across all words
        and subjects. This avoids reloading 3GB model weights 24 times.

    Returns:
        DataFrame with one row per classified word, or None if the
        subject could not be processed (no audio, no clips, etc.).
    """

    print(f"\n{'=' * 60}")
    print(f"  Subject {sub_id}")
    print(f"{'=' * 60}")

    # ---- Stage A: Step 1 — Resolve audio file ----
    raw_audio_path = resolve_audio_path(sub_id, sub_path, TEXT)
    if raw_audio_path is None:
        print(f"  [{sub_id}] No audio found — skipping.")
        return None

    # ---- Stage A: Step 2 — Noise removal ----
    clean_audio_path = clean_audio_pipeline(sub_id, raw_audio_path)

    # ---- Stage A: Step 3 — WhisperX transcription + forced alignment ----
    # If timestamps Excel already exists (from a previous run), skip this step.
    timestamps_path = find_timestamps_file(sub_path, sub_id, TEXT)

    if timestamps_path is None:
        print(f"  [{sub_id}] No timestamps file — running feature_pipeline...")
        should_skip = feature_pipeline(
            sub_id,
            TEXT,
            sub_path,
            clean_audio_path,
            whisperx_device,
            whisperx_model,
            whisperx_align_model,
            whisperx_metadata,
            expected_words,
        )
        if should_skip:
            print(f"  [{sub_id}] feature_pipeline returned skip — skipping subject.")
            return None

        timestamps_path = find_timestamps_file(sub_path, sub_id, TEXT)
        if timestamps_path is None:
            print(f"  [{sub_id}] Still no timestamps after feature_pipeline — skipping.")
            return None
    else:
        print(f"  [{sub_id}] Timestamps already exist — skipping transcription.")

    # ---- Stage A: Step 4 — Slice audio into word clips ----
    clips_dir = os.path.join(sub_path, f"{TEXT}_word_clips_{sub_id}")

    if os.path.exists(clips_dir) and len(os.listdir(clips_dir)) > 0:
        # Clips already exist from a previous run — reuse them
        print(f"  [{sub_id}] Clips already exist — skipping slicing.")
    else:
        print(f"  [{sub_id}] Slicing audio into word clips...")
        slice_audio(
            sub_id=sub_id,
            text=TEXT,
            audio_path=clean_audio_path,
            timestamps_path=timestamps_path,
            padding_start=PADDING_START,
            padding_end=PADDING_END,
        )

    # ---- Stage B: Classify each word clip ----
    clip_files = sorted([f for f in os.listdir(clips_dir) if f.endswith(".wav")])

    if len(clip_files) == 0:
        print(f"  [{sub_id}] No clips found in {clips_dir} — skipping.")
        return None

    print(f"  [{sub_id}] Classifying {len(clip_files)} word clips...")

    rows = []

    for clip_file in clip_files:
        word = extract_word_from_clip_filename(clip_file)

        # Skip clips for words outside our 24-word target list
        if word not in TARGET_WORDS:
            print(f"  [{sub_id}] Skipping unknown word: {word}")
            continue

        model_path = get_model_path(word)

        if not os.path.exists(model_path):
            # No model trained for this word yet — mark as no_model
            print(f"  [{sub_id}] No model for '{word}' at {model_path} — skipping.")
            rows.append({
                "subject_id":          sub_id,
                "word":                word,
                "final_decision":      "no_model",
                "probability_correct": None,
                "phonetic_score":      None,
                "decision_reason":     "no_model_found",
                "oneclass_score":      None,
            })
            continue

        # Get the correct wav2vec2 model for this word.
        # Load it into cache if not already loaded — each model loads once.
        word_model_name = WAV2VEC2_MODEL_PER_WORD.get(word, WAV2VEC2_MODEL_NAME)
        if word_model_name not in wav2vec2_cache:
            print(f"  Loading wav2vec2: {word_model_name}")
            wav2vec2_cache[word_model_name] = load_wav2vec2(word_model_name)
        processor, wav2vec2_model_loaded = wav2vec2_cache[word_model_name]

        clip_path = os.path.join(clips_dir, clip_file)

        try:
            result = predict_one_audio(
                model_path=model_path,
                audio_path=clip_path,
                target_word=word,
                subject_id=str(sub_id),
                processor=processor,
                wav2vec2_model=wav2vec2_model_loaded,
            )

            rows.append({
                "subject_id":               sub_id,
                "word":                     word,
                "final_decision":           result["final_decision"],
                "model_prediction":         result["model_prediction"],
                "probability_correct":      result["probability_correct"],
                "phonetic_score":           result["phonetic_score"],
                "oneclass_score":           result["oneclass_score"],
                "decision_reason":          result["decision_reason"],
                "matched_transcription":    result.get("matched_transcription"),
                "transcription_exact_match": result.get("transcription_exact_match"),
            })

            print(f"  [{sub_id}] {word:<25} -> {result['final_decision']}")

        except Exception as e:
            # Log the error and continue with remaining words
            print(f"  [{sub_id}] ERROR classifying {word}: {e}")
            rows.append({
                "subject_id":          sub_id,
                "word":                word,
                "final_decision":      "error",
                "probability_correct": None,
                "phonetic_score":      None,
                "decision_reason":     f"error: {str(e)}",
                "oneclass_score":      None,
            })

    if not rows:
        return None

    # Save per-subject predictions CSV
    results_df  = pd.DataFrame(rows)
    output_path = os.path.join(sub_path, f"predictions_{sub_id}.csv")
    results_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n  [{sub_id}] Saved: {output_path}")

    return results_df


# ----------------------------------------------------------------
# MAIN ENTRY POINT
# ----------------------------------------------------------------

def main():
    """
    Processes a list of new subjects through the full pipeline.

    Before running:
        - Add subject IDs to the subject_ids list below.
        - Each subject must have a folder at:
              folder1/folder2/new_subjects/subject_{id}/

    Loads once and reuses across all subjects:
        - WhisperX model (large ASR + alignment model)
        - Expected words list (from text file)
        - wav2vec2 models (via cache dict — one entry per model name)

    Saves:
        - Per-subject: subject_{id}/predictions_{id}.csv
        - Combined:    new_subjects/all_predictions.csv
    """

    # ---- Settings ----
    subject_ids = [1904, 1908, 1957, 1988]   # <- add new subject IDs here

    new_subjects_base = os.path.join(gv.folder1, gv.folder2, "new_subjects")

    # ---- Load WhisperX once — shared across all subjects ----
    print("Loading WhisperX model...")
    device, whisperx_model, align_model, metadata = load_whisperx_model()

    # ---- Load expected words from text file ----
    text_path = os.path.join(gv.folder1, f"texts/{TEXT}_1920/alltext.txt")
    with open(text_path, encoding="utf-8") as f:
        expected_words = f.read().split()
    expected_words = expected_words[:24]

    # ---- wav2vec2 cache — each model name maps to (processor, model) ----
    # Pre-load the default model eagerly so the first word doesn't wait.
    # Additional models (e.g. skylord) are loaded on demand when first needed.
    wav2vec2_cache = {}
    print(f"Loading default wav2vec2 model: {WAV2VEC2_MODEL_NAME}")
    wav2vec2_cache[WAV2VEC2_MODEL_NAME] = load_wav2vec2(WAV2VEC2_MODEL_NAME)

    # ---- Process each subject sequentially ----
    all_results = []

    for sub_id in subject_ids:
        sub_path = os.path.join(new_subjects_base, f"subject_{sub_id}")

        if not os.path.exists(sub_path):
            print(f"Subject folder not found: {sub_path} — skipping.")
            continue

        result_df = predict_subject(
            sub_id=sub_id,
            sub_path=sub_path,
            whisperx_device=device,
            whisperx_model=whisperx_model,
            whisperx_align_model=align_model,
            whisperx_metadata=metadata,
            expected_words=expected_words,
            wav2vec2_cache=wav2vec2_cache,
        )

        if result_df is not None:
            all_results.append(result_df)

    # ---- Save combined results across all subjects ----
    if all_results:
        combined_df   = pd.concat(all_results, ignore_index=True)
        combined_path = os.path.join(new_subjects_base, "all_predictions.csv")
        combined_df.to_csv(combined_path, index=False, encoding="utf-8-sig")

        print(f"\n{'=' * 60}")
        print("  ALL SUBJECTS FINISHED")
        print(f"{'=' * 60}\n")
        print(f"Saved combined results: {combined_path}\n")

        # Print summary table
        print(f"{'Subject':<12} {'Word':<25} {'Decision':<20} {'Prob':>6}")
        print("-" * 68)
        score_correct = 0
        for _, row in combined_df.iterrows():
            prob = f"{row['probability_correct']:.3f}" if pd.notna(row['probability_correct']) else "N/A"
            print(f"{str(row['subject_id']):<12} {row['word']:<25} {row['final_decision']:<20} {prob:>6}")
            if row['final_decision'] in ('correct', 'possible_correct'):
                score_correct += 1

        print(f"\nTotal correct predictions: {score_correct} / {len(combined_df)}\n")


if __name__ == "__main__":
    main()