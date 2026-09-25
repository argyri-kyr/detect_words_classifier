# predict_new_subjects_scan.py
#
# Alternative pipeline for new subjects that does NOT use WhisperX.
# Instead, uses the per-word SVC models to both LOCATE and CLASSIFY
# each word directly from the continuous audio.
#
# Why:
#     WhisperX + forced alignment sometimes gives wrong timestamps
#     for certain subjects (e.g. off-script speech, heavy accents,
#     slow readers). This pipeline bypasses WhisperX entirely and
#     uses the trained per-word SVCs as word detectors.
#
# How it works:
#     1. Load full audio, encode once with wav2vec2
#     2. For each of the 24 words IN ORDER, slide a window through
#        the audio from the current cursor position
#     3. The window position with the highest SVC "correct" probability
#        is where the word most likely occurs
#     4. The SVC probability at that position is also the classification
#        result — so localization and classification happen in one pass
#
# Usage:
#     python predict_new_subjects_scan.py

import os
import sys
import inspect
import joblib
import numpy as np
import pandas as pd
import torch
import librosa


# ----------------------------------------------------------------
# Path setup — import from classifier project
# ----------------------------------------------------------------
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, currentdir)

import globalVars as gv
from globalVars import (
    folder1,
    text,
    WAV2VEC2_MODEL_NAME,
    EMBEDDING_LAYER_PER_WORD,
    USE_PHONETIC_SCORE,
)
from wav2vec2_features import load_wav2vec2
from audio_utils import load_and_preprocess_audio
from decision_rules import make_final_decision


# ----------------------------------------------------------------
# Settings
# ----------------------------------------------------------------
TEXT = "decodewordspseudo"

# The 24 target words in reading order.
# Load from text
with open(folder1 + f'texts/{text}_1920/alltext.txt', encoding="utf-8") as f:
    TARGET_WORDS = f.read().split()

TARGET_WORDS = TARGET_WORDS[:24]

# Scanning parameters
SAMPLE_RATE = 16000

# Typical word durations in seconds — used to size the sliding window.
# Most Greek words in this list are 0.5–1.5s long.
# We use a range of window sizes and pick the best score.
WINDOW_DURATIONS = [0.6, 0.8, 1.0, 1.2, 1.5]

# How far ahead (in seconds) from the cursor to search for each word.
# 25s handles pauses, retries, off-script speech between words.
LOOKAHEAD_SECONDS = 25.0

# Step size for sliding the window (in seconds).
# Smaller = more precise but slower.
STEP_SECONDS = 0.15

# Minimum SVC probability to consider a window a valid detection.
# Below this, the word is marked as "not_found".
MIN_DETECTION_CONFIDENCE = 0.35

# After detecting a word, advance cursor by at least this many seconds
# to avoid re-detecting the same audio region for the next word.
MIN_CURSOR_ADVANCE = 0.3


def get_model_path(word):
    """
    Returns the path to model.joblib for a given word.
    """
    safe_model_name = WAV2VEC2_MODEL_NAME.replace("/", "__")
    layer = EMBEDDING_LAYER_PER_WORD.get(word, gv.EMBEDDING_LAYER)
    safe_word = word.replace(" ", "_")

    return os.path.join(
        "outputs",
        safe_model_name,
        f"layer_{layer}",
        safe_word,
        "model.joblib",
    )


def encode_full_audio(audio_path, processor, wav2vec2_model):
    """
    Encode the full audio file with wav2vec2 and return hidden states
    for all layers.

    Returns:
        hidden_states: list of tensors, one per layer
        duration_seconds: float, total audio duration
    """
    waveform, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    duration_seconds = len(waveform) / sr

    print(f"  Audio duration: {duration_seconds:.1f}s ({len(waveform)} samples)")

    # Encode with wav2vec2
    inputs = processor(
        waveform,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )

    with torch.no_grad():
        outputs = wav2vec2_model(**inputs, output_hidden_states=True)

    # hidden_states is a tuple of (num_layers+1) tensors,
    # each of shape (1, num_frames, hidden_dim)
    return outputs.hidden_states, duration_seconds, waveform


def extract_window_embedding(hidden_states, layer, start_frame, end_frame):
    """
    Extract mean+std embedding for a window of frames from a specific layer.
    Same aggregation as extract_wav2vec2_embedding() in wav2vec2_features.py.
    """
    layer_output = hidden_states[layer]  # (1, total_frames, hidden_dim)
    window = layer_output[0, start_frame:end_frame, :]  # (window_frames, hidden_dim)

    if window.shape[0] == 0:
        # Empty window — return zeros
        hidden_dim = layer_output.shape[2]
        return np.zeros(hidden_dim * 2)

    mean_vec = window.mean(dim=0).numpy()
    std_vec = window.std(dim=0).numpy()
    embedding = np.concatenate([mean_vec, std_vec])

    return embedding


def seconds_to_frame(seconds, total_samples, total_frames):
    """
    Convert a time in seconds to the corresponding wav2vec2 frame index.
    wav2vec2 downsamples audio: ~320 samples per frame at 16kHz = 50 frames/sec.
    """
    sample_idx = int(seconds * SAMPLE_RATE)
    sample_idx = min(sample_idx, total_samples - 1)
    frame_idx = int(sample_idx * total_frames / total_samples)
    return max(0, min(frame_idx, total_frames - 1))


def scan_for_word(
    word,
    model_data,
    hidden_states,
    cursor_seconds,
    total_samples,
    total_frames,
    duration_seconds,
):
    """
    Scan the audio from cursor_seconds forward to find the best window
    for a given word using its trained SVC model.

    Returns:
        best_start_s: float — start time of best window (seconds)
        best_end_s: float — end time of best window (seconds)
        best_prob: float — SVC probability_correct at best window
        best_embedding: np.array — the embedding at the best window
    """
    model = model_data["model"]
    feature_columns = model_data["feature_columns"]
    layer = model_data.get("embedding_layer", gv.EMBEDDING_LAYER)

    # Number of features the model expects
    n_features = len(feature_columns)

    # How many of those are wav2vec2 embedding dimensions
    # (the rest are phonetic_score, whisper_similarity, etc.)
    n_embedding_dims = n_features
    if USE_PHONETIC_SCORE:
        n_embedding_dims -= 1  # last feature is phonetic_score

    max_search_time = min(
        cursor_seconds + LOOKAHEAD_SECONDS,
        duration_seconds,
    )

    best_prob = -1.0
    best_start_s = cursor_seconds
    best_end_s = cursor_seconds + 1.0
    best_embedding = None

    for win_dur in WINDOW_DURATIONS:
        t = cursor_seconds

        while t + win_dur <= max_search_time:
            start_frame = seconds_to_frame(t, total_samples, total_frames)
            end_frame = seconds_to_frame(t + win_dur, total_samples, total_frames)

            if end_frame <= start_frame:
                t += STEP_SECONDS
                continue

            embedding = extract_window_embedding(hidden_states, layer, start_frame, end_frame)

            # Build feature vector matching what the model expects
            if len(embedding) > n_embedding_dims:
                embedding = embedding[:n_embedding_dims]
            elif len(embedding) < n_embedding_dims:
                # Pad with zeros if embedding is shorter than expected
                embedding = np.concatenate([
                    embedding,
                    np.zeros(n_embedding_dims - len(embedding)),
                ])

            # Add phonetic_score placeholder if needed
            # We don't have a real phonetic score during scanning,
            # so use a neutral value (100 = perfect match assumed).
            features = embedding.copy()
            if USE_PHONETIC_SCORE:
                features = np.append(features, 100.0)

            # Predict
            features_2d = features.reshape(1, -1)

            try:
                prob = model.predict_proba(features_2d)[0]
                # prob[1] = probability of class "correct" (label_id=1)
                prob_correct = prob[1] if len(prob) > 1 else prob[0]
            except Exception:
                t += STEP_SECONDS
                continue

            if prob_correct > best_prob:
                best_prob = prob_correct
                best_start_s = t
                best_end_s = t + win_dur
                best_embedding = features

            t += STEP_SECONDS

    return best_start_s, best_end_s, best_prob, best_embedding


def scan_and_classify_subject(
    sub_id,
    audio_path,
    target_words,
    processor,
    wav2vec2_model_loaded,
):
    """
    Scan the full audio for all 24 words in order.
    Returns a DataFrame with timestamps and classification results.
    """
    print(f"\n{'=' * 60}")
    print(f"  Subject {sub_id} — Scan-based detection")
    print(f"{'=' * 60}")

    # ----------------------------------------------------------------
    # Step 1: Encode full audio once with wav2vec2
    # ----------------------------------------------------------------
    print("  Encoding full audio with wav2vec2...")
    hidden_states, duration_seconds, waveform = encode_full_audio(
        audio_path, processor, wav2vec2_model_loaded,
    )

    total_samples = len(waveform)
    total_frames = hidden_states[0].shape[1]  # frames in layer 0
    print(f"  Total frames: {total_frames} ({total_frames / 50:.1f}s at 50 fps)")

    # ----------------------------------------------------------------
    # Step 2: Scan for each word in order
    # ----------------------------------------------------------------
    cursor_seconds = 0.0
    rows = []

    for word_idx, word in enumerate(target_words):
        model_path = get_model_path(word)

        if not os.path.exists(model_path):
            print(f"  [{sub_id}] ⚠️  No model for '{word}' — skipping")
            rows.append({
                "subject_id": sub_id,
                "word": word,
                "word_index": word_idx,
                "start_s": None,
                "end_s": None,
                "final_decision": "no_model",
                "probability_correct": None,
                "decision_reason": "no_model_found",
            })
            continue

        # Load word model
        model_data = joblib.load(model_path)

        # Override global embedding layer for this word
        layer = EMBEDDING_LAYER_PER_WORD.get(word, gv.EMBEDDING_LAYER)
        original_layer = gv.EMBEDDING_LAYER
        gv.EMBEDDING_LAYER = layer

        try:
            best_start, best_end, best_prob, best_embedding = scan_for_word(
                word=word,
                model_data=model_data,
                hidden_states=hidden_states,
                cursor_seconds=cursor_seconds,
                total_samples=total_samples,
                total_frames=total_frames,
                duration_seconds=duration_seconds,
            )
        finally:
            gv.EMBEDDING_LAYER = original_layer

        # ----------------------------------------------------------------
        # Step 3: Classify using decision rules
        # ----------------------------------------------------------------
        if best_prob < MIN_DETECTION_CONFIDENCE:
            # Word not confidently found — mark as not found
            print(f"  [{sub_id}] {word:<25} ❌ not found (best prob={best_prob:.3f})")
            rows.append({
                "subject_id": sub_id,
                "word": word,
                "word_index": word_idx,
                "start_s": best_start,
                "end_s": best_end,
                "final_decision": "not_found",
                "probability_correct": best_prob,
                "decision_reason": f"below_detection_threshold_{MIN_DETECTION_CONFIDENCE}",
            })
            # Still advance cursor a bit so we don't get stuck
            cursor_seconds = best_start + MIN_CURSOR_ADVANCE
            continue

        # Apply decision rules for the final classification
        best_threshold = model_data.get("best_threshold", 0.5)
        best_phonetic_cutoff = model_data.get("best_phonetic_cutoff", None)

        decision = make_final_decision(
            target_word=word,
            probability_correct=best_prob,
            phonetic_score=100.0,  # no real phonetic score in scan mode
            probability_threshold=best_threshold,
            phonetic_cutoff=best_phonetic_cutoff,
            matched_transcription=None,
            detected_status="found",
            whisper_similarity=None,
            whisper_transcription=None,
        )

        final_decision = decision["final_decision"]
        decision_reason = decision["decision_reason"]

        status_icon = "✅" if final_decision == "correct" else "⚠️"
        print(
            f"  [{sub_id}] {word:<25} {status_icon} {final_decision:<20} "
            f"prob={best_prob:.3f}  t=[{best_start:.2f}–{best_end:.2f}]"
        )

        rows.append({
            "subject_id": sub_id,
            "word": word,
            "word_index": word_idx,
            "start_s": round(best_start, 3),
            "end_s": round(best_end, 3),
            "final_decision": final_decision,
            "model_prediction": decision.get("model_prediction"),
            "probability_correct": round(best_prob, 4),
            "decision_reason": decision_reason,
            "transcription_exact_match": decision.get("transcription_exact_match"),
            "review_required": decision.get("review_required"),
        })

        # Advance cursor past this word
        cursor_seconds = best_end + MIN_CURSOR_ADVANCE

    return pd.DataFrame(rows)


def main():
    # ----------------------------------------------------------------
    # Settings
    # ----------------------------------------------------------------
    subject_ids = [1973, 1976, 1984]  

    new_subjects_base = os.path.join(
        gv.folder1,
        gv.folder2,
        "new_subjects",
    )

    # ----------------------------------------------------------------
    # Load wav2vec2 once
    # ----------------------------------------------------------------
    print("Loading wav2vec2 model...")
    processor, wav2vec2_model_loaded = load_wav2vec2(WAV2VEC2_MODEL_NAME)

    # ----------------------------------------------------------------
    # Process each subject
    # ----------------------------------------------------------------
    all_results = []

    for sub_id in subject_ids:
        sub_path = os.path.join(new_subjects_base, f"subject_{sub_id}")

        if not os.path.exists(sub_path):
            print(f"⚠️  Subject folder not found: {sub_path} — skipping.")
            continue

        # Find the audio file (clean version preferred)
        # Look for the cleaned audio first, then fall back to raw
        audio_path = None
        for candidate in [
            os.path.join(sub_path, f"{TEXT}_clean_{sub_id}.wav"),
            os.path.join(sub_path, f"{TEXT}_{sub_id}.wav"),
            os.path.join(sub_path, f"{sub_id}.wav"),
        ]:
            if os.path.exists(candidate):
                audio_path = candidate
                break

        if audio_path is None:
            # Try to find any .wav file
            wav_files = [f for f in os.listdir(sub_path) if f.endswith(".wav")]
            if wav_files:
                audio_path = os.path.join(sub_path, wav_files[0])

        if audio_path is None:
            print(f"⚠️  No audio found for subject {sub_id} — skipping.")
            continue

        print(f"  Using audio: {audio_path}")

        result_df = scan_and_classify_subject(
            sub_id=sub_id,
            audio_path=audio_path,
            target_words=TARGET_WORDS,
            processor=processor,
            wav2vec2_model_loaded=wav2vec2_model_loaded,
        )

        if result_df is not None and len(result_df) > 0:
            # Save per-subject predictions
            output_path = os.path.join(sub_path, f"predictions_scan_{sub_id}.csv")
            result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
            print(f"\n  Saved: {output_path}")

            all_results.append(result_df)

    # ----------------------------------------------------------------
    # Combined results
    # ----------------------------------------------------------------
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_path = os.path.join(new_subjects_base, "all_predictions_scan.csv")
        combined_df.to_csv(combined_path, index=False, encoding="utf-8-sig")

        print(f"\n{'=' * 60}")
        print("  ALL SUBJECTS FINISHED (scan mode)")
        print(f"{'=' * 60}\n")
        print(f"Saved: {combined_path}")
        print()

        # Summary table
        print(f"{'Subject':<10} {'Word':<25} {'Decision':<20} {'Prob':>6}  {'Time':>12}")
        print("-" * 78)
        for _, row in combined_df.iterrows():
            prob = f"{row['probability_correct']:.3f}" if pd.notna(row['probability_correct']) else "N/A"
            t_start = f"{row['start_s']:.2f}" if pd.notna(row.get('start_s')) else "?"
            t_end = f"{row['end_s']:.2f}" if pd.notna(row.get('end_s')) else "?"
            print(
                f"{str(row['subject_id']):<10} {row['word']:<25} "
                f"{row['final_decision']:<20} {prob:>6}  [{t_start}–{t_end}]"
            )


if __name__ == "__main__":
    main()