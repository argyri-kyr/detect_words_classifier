# continuous_scan.py
#
# Independent, standalone alternative to the clip-slicing pipeline
# (predict.py / predict_new_subjects_main.py). Does NOT modify or
# depend on predict.py in any way — built so results can be compared
# side-by-side against the existing isolated-clip approach.
#
# Why this exists:
# WhisperX forced-alignment can misalign downstream words when a child
# says off-script speech ("I don't know this word", "well done, keep
# going"). When that happens, slice_audio() cuts clips at the wrong
# boundaries and predict_one_audio() classifies garbage with no way
# to know the clip itself was wrong.
#
# This scanner encodes the full recording once and searches forward
# for each target word in its known fixed order, never depending on
# WhisperX/forced-alignment succeeding. Off-script speech is naturally
# skipped over since it scores low against every word's own classifier.
#
# Multi-layer support (layer_combined words):
# Words trained with EMBEDDING_LAYERS_COMBINE_PER_WORD use embeddings
# from multiple layers concatenated. The scanner handles this by running
# separate forward passes per layer, slicing each layer's sequence at the
# same window boundaries, and concatenating the pooled vectors before
# feeding to the SVC. This matches exactly what extract_features.py does
# during training, so the feature dimensions are always consistent.

import os
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import joblib
import librosa
import torch

import globalVars as gv
from audio_utils import load_and_preprocess_audio
from wav2vec2_features import load_wav2vec2
from decision_rules import make_final_decision
from globalVars import EMBEDDING_LAYERS_COMBINE_PER_WORD, WAV2VEC2_MODEL_PER_WORD


# ============================================================
# Settings
# ============================================================
FRAME_STRIDE_S = 0.02                      # wav2vec2 frame rate (~20ms)
SCAN_STRIDE_FRAMES = 3                     # window step size (~60ms)
WINDOW_SIZE_MULTIPLIERS = [0.8, 1.0, 1.3]  # candidate window sizes per word
MIN_CONFIDENCE_TO_ACCEPT_WINDOW = 0.35     # below this -> "no match found"
                                            # lower than isolated-clip thresholds
                                            # since continuous-audio windows carry
                                            # coarticulation blur from neighboring
                                            # speech — tune against ground truth
DEFAULT_LOOKAHEAD_S = 25.0                 # how far past cursor to search
DEFAULT_PROBABILITY_THRESHOLD = 0.50       # trained thresholds assume real
                                            # phonetic_score; not reused here

DURATION_CACHE_PATH = os.path.join("outputs", "word_expected_durations_scan.json")


# ============================================================
# WAV2VEC2 SEQUENCE EXTRACTION
# ============================================================

def extract_wav2vec2_sequence_single(audio, processor, model, layer):
    """
    Extract full frame-level wav2vec2 hidden states for ONE layer,
    without mean+std pooling, so the caller can slice/window first.

    Returns:
        numpy array of shape (T, hidden_size) where T = number of frames.
        Each row is one 20ms time frame.

    Why no pooling here:
        The scanner needs to slide windows across the sequence and pool
        each window separately. Pooling the whole sequence first would
        lose the time dimension needed for windowing.
    """

    inputs = processor(
        audio,
        sampling_rate=gv.SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden = outputs.hidden_states[layer].squeeze(0)
    return hidden.numpy()


def extract_wav2vec2_sequences_multilayer(audio, processor, model, layers):
    """
    Extract frame-level hidden states for MULTIPLE layers in one forward pass.

    Runs wav2vec2 once and returns a dict mapping layer_index -> (T, hidden_size).
    One forward pass is enough since all hidden states are returned together.

    Why one forward pass:
        output_hidden_states=True returns ALL layers at once. Extracting
        multiple layers in one pass is identical in cost to extracting one.

    Returns:
        dict: {layer_index: numpy array (T, hidden_size)}
    """

    inputs = processor(
        audio,
        sampling_rate=gv.SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    return {
        layer: outputs.hidden_states[layer].squeeze(0).numpy()
        for layer in layers
    }


def pool_window_single(sequence_window):
    """
    Mean+std pool a window slice from a single-layer sequence.

    Matches the pooling in extract_wav2vec2_embedding() exactly:
        mean over time → hidden_size values
        std  over time → hidden_size values
        concatenate    → 2 * hidden_size values (2048 for large models)

    Input:
        sequence_window: numpy array (window_frames, hidden_size)

    Returns:
        1D numpy array (2 * hidden_size,)
    """

    mean_vector = sequence_window.mean(axis=0)
    std_vector  = sequence_window.std(axis=0)
    return np.concatenate([mean_vector, std_vector])


def pool_window_multilayer(sequences_by_layer, start, end, layers):
    """
    Pool window [start:end] from multiple layer sequences and concatenate.

    Matches extract_wav2vec2_embedding_multilayer() exactly:
        for each layer: mean+std pool the window slice → 2048 values
        concatenate all layers → len(layers) * 2048 values

    After PCA (inside the sklearn Pipeline in model.joblib), this is
    compressed to ~50-150 components before the SVC sees it.

    Args:
        sequences_by_layer: dict {layer_index: (T, hidden_size) array}
        start:  window start frame index
        end:    window end frame index
        layers: list of layer indices to use (same as training)

    Returns:
        1D numpy array (len(layers) * 2 * hidden_size,)
    """

    pooled_parts = []
    for layer in layers:
        window = sequences_by_layer[layer][start:end]
        pooled_parts.append(pool_window_single(window))
    return np.concatenate(pooled_parts)


# ============================================================
# EXPECTED WORD DURATION (for window sizing)
# ============================================================

def get_word_expected_duration(word, metadata_csv):
    """
    Median duration (seconds) of correct training clips for one word.
    Cached to disk so it is only computed once per word.

    Used to set the expected window size during scanning — windows are
    sized at 0.8x, 1.0x, and 1.3x of the median correct clip duration
    to account for fast and slow readers.

    Uses librosa.load() + len/sr instead of librosa.get_duration(path=...)
    since the path= kwarg is not supported across all librosa versions.
    """

    cache = {}
    if os.path.exists(DURATION_CACHE_PATH):
        with open(DURATION_CACHE_PATH, "r", encoding="utf-8") as f:
            cache = json.load(f)

    if word in cache:
        return cache[word]

    metadata = pd.read_csv(metadata_csv)
    correct_rows = metadata[
        (metadata["target_word"] == word) & (metadata["label"] == "correct")
    ]

    if len(correct_rows) == 0:
        raise ValueError(f"No correct training clips found for word: {word}")

    durations = []
    for path in correct_rows["filepath"]:
        audio, sr = librosa.load(path, sr=None, mono=True)
        durations.append(len(audio) / sr)

    median_duration = float(np.median(durations))

    cache[word] = median_duration
    os.makedirs(os.path.dirname(DURATION_CACHE_PATH), exist_ok=True)
    with open(DURATION_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    return median_duration


# ============================================================
# TYPICAL PHONETIC SCORE (for feature-vector fallback)
# ============================================================

def get_word_typical_phonetic_score(word, features_csv):
    """
    Median phonetic_score among correct-labeled training samples for one word.

    Used as a neutral fallback feature value during scanning, since no real
    per-window phonetic score is available without forced alignment.

    Why not use 0:
        Using 0 (maximally bad score) systematically depresses
        probability_correct even for genuinely correct windows, since the
        model was trained with phonetic_score as a real signal and correct
        samples almost always score 80-100. The median of correct training
        scores is a neutral, non-punishing fallback.
    """

    features_df = pd.read_csv(features_csv)
    correct_scores = features_df[
        features_df["label"] == "correct"
    ]["phonetic_score"].dropna()

    if len(correct_scores) == 0:
        return 90.0  # conservative fallback if data is missing

    return float(correct_scores.median())


# ============================================================
# CORE SCAN: find best-scoring window for one word
# ============================================================

def scan_word_in_sequence(
    word,
    sequences_by_layer,
    model_path,
    cursor_frame,
    max_lookahead_frames,
    window_frames_options,
    phonetic_score_fallback,
):
    """
    Slides windows across [cursor_frame, cursor_frame + max_lookahead_frames],
    scores each with the word's trained SVC, and returns the single
    best-scoring window plus its one-class gate score.

    Supports both single-layer and combined-layer words:
        - Single-layer: pools one layer's sequence at each window
        - Combined-layer: pools multiple layers and concatenates
          (matching the training feature vector exactly)

    PCA is inside the sklearn Pipeline in model.joblib — it is applied
    automatically by classifier.predict_proba(), no explicit step needed.

    Returns None if nothing clears MIN_CONFIDENCE_TO_ACCEPT_WINDOW.

    Args:
        word:                  target word (used to look up layer config)
        sequences_by_layer:    dict {layer_index: (T, hidden_size) array}
        model_path:            path to model.joblib for this word
        cursor_frame:          frame index to start searching from
        max_lookahead_frames:  how many frames ahead to search
        window_frames_options: list of window sizes in frames to try
        phonetic_score_fallback: neutral phonetic score used as feature
    """

    model_data       = joblib.load(model_path)
    classifier       = model_data["model"]
    feature_columns  = model_data["feature_columns"]

    # Load one-class gate if available
    ocsvm_path = os.path.join(os.path.dirname(model_path), "model_oneclass.joblib")
    ocsvm_model            = None
    ocsvm_feature_columns  = None
    if os.path.exists(ocsvm_path):
        ocsvm_data            = joblib.load(ocsvm_path)
        ocsvm_model           = ocsvm_data["model"]
        ocsvm_feature_columns = ocsvm_data["feature_columns"]

    # Determine whether this word uses combined layers or single layer
    is_combined = word in EMBEDDING_LAYERS_COMBINE_PER_WORD
    layers      = EMBEDDING_LAYERS_COMBINE_PER_WORD[word] if is_combined else None

    # Use any layer's sequence length for boundary calculation
    any_layer    = next(iter(sequences_by_layer))
    seq_len      = sequences_by_layer[any_layer].shape[0]
    search_end   = min(cursor_frame + max_lookahead_frames, seq_len)

    best_probability = -1.0
    best_result      = None

    for window_frames in window_frames_options:
        window_frames = max(window_frames, 1)

        for start in range(cursor_frame, search_end - window_frames, SCAN_STRIDE_FRAMES):
            end = start + window_frames

            # Pool the window — single or multi-layer depending on word config
            if is_combined:
                pooled = pool_window_multilayer(sequences_by_layer, start, end, layers)
            else:
                single_layer = next(iter(sequences_by_layer))
                pooled = pool_window_single(sequences_by_layer[single_layer][start:end])

            # Build feature vector matching training order:
            #   wav2vec2 embedding (single or combined layers)
            #   + phonetic_score fallback (if USE_PHONETIC_SCORE)
            features = pooled
            if gv.USE_PHONETIC_SCORE:
                features = np.append(features, phonetic_score_fallback)

            if len(features) != len(feature_columns):
                # Dimension mismatch — skip this window silently.
                # Should not happen if globalVars matches training config.
                continue

            row = {col: val for col, val in zip(feature_columns, features)}
            X   = pd.DataFrame([row])

            # PCA + scaler applied inside Pipeline automatically
            probability = float(classifier.predict_proba(X)[0][1])

            if probability > best_probability:
                best_probability = probability

                oneclass_score = None
                if ocsvm_model is not None:
                    X_oc = X[ocsvm_feature_columns]
                    oneclass_score = float(ocsvm_model.decision_function(X_oc)[0])

                best_result = {
                    "probability_correct":  probability,
                    "oneclass_score":       oneclass_score,
                    "window_start_frame":   start,
                    "window_end_frame":     end,
                }

    if best_result is None or best_result["probability_correct"] < MIN_CONFIDENCE_TO_ACCEPT_WINDOW:
        return None

    return best_result


# ============================================================
# TOP-LEVEL ENTRY POINT
# ============================================================

def predict_continuous_audio(
    audio_path,
    words_in_order,
    get_model_path_fn,
    get_metadata_csv_fn,
    get_features_csv_fn,
    subject_id,
    processor=None,
    wav2vec2_model=None,
):
    """
    Classify all target words directly from one continuous recording,
    without slicing physical clips and without depending on WhisperX
    alignment succeeding.

    How it works:
        1. Load and preprocess the full audio once.
        2. For each unique (model_name, layer) combination used across
           all words, run one wav2vec2 forward pass to get frame-level
           hidden states. Combined-layer words get all their layers in
           one pass since output_hidden_states=True returns everything.
        3. For each word in reading order, slide windows of candidate
           sizes across the sequence starting from the current cursor.
           Score each window with the word's SVC (PCA inside Pipeline).
        4. Accept the best-scoring window if it clears the confidence
           threshold. Advance cursor to end of accepted window.
        5. If no window clears, flag as no_match_found and advance
           cursor by the word's expected duration.

    Per-word model and layer config:
        Each word uses its assigned wav2vec2 model (WAV2VEC2_MODEL_PER_WORD)
        and layer config (EMBEDDING_LAYERS_COMBINE_PER_WORD or single layer).
        Forward passes are grouped by (model_name, layers) so each unique
        combination runs only once regardless of how many words share it.

    Args:
        audio_path:          path to the continuous recording .wav
        words_in_order:      target words in expected reading order
        get_model_path_fn:   word -> path to model.joblib
        get_metadata_csv_fn: word -> path to training metadata.csv
                             (used to estimate expected clip duration)
        get_features_csv_fn: word -> path to training features.csv
                             (used to compute neutral phonetic_score fallback)
        subject_id:          child's ID, added to each result row
        processor:           pre-loaded wav2vec2 processor (optional)
        wav2vec2_model:      pre-loaded wav2vec2 model (optional)
                             If not provided, loads from WAV2VEC2_MODEL_NAME.
                             For batch runs pass pre-loaded models to avoid
                             reloading wav2vec2 per subject.

    Returns:
        List of per-word result dicts (one per word in words_in_order).
    """

    audio = load_and_preprocess_audio(audio_path)

    # ----------------------------------------------------------------
    # Group words by (model_name, layer_key) to minimise forward passes.
    # Single-layer words: key = (model_name, layer_index)
    # Combined-layer words: key = (model_name, tuple(layers))
    # ----------------------------------------------------------------
    # We need one forward pass per unique (model_name, layer_set) combo.
    # Since output_hidden_states=True returns all layers at once, combined-
    # layer words only need one pass regardless of how many layers they use.

    # Map each word to its (model_name, layer_key, layers_needed)
    word_config = {}
    for word in words_in_order:
        model_name = WAV2VEC2_MODEL_PER_WORD.get(word, gv.WAV2VEC2_MODEL_NAME)
        if word in EMBEDDING_LAYERS_COMBINE_PER_WORD:
            layers     = EMBEDDING_LAYERS_COMBINE_PER_WORD[word]
            layer_key  = (model_name, tuple(layers))
        else:
            layer      = gv.EMBEDDING_LAYER_PER_WORD.get(word, gv.EMBEDDING_LAYER)
            layers     = [layer]
            layer_key  = (model_name, layer)
        word_config[word] = {
            "model_name": model_name,
            "layer_key":  layer_key,
            "layers":     layers,
            "is_combined": word in EMBEDDING_LAYERS_COMBINE_PER_WORD,
        }

    # Build set of unique (model_name, layers_needed) combos
    unique_combos = {}
    for word, cfg in word_config.items():
        key = cfg["layer_key"]
        if key not in unique_combos:
            unique_combos[key] = {
                "model_name": cfg["model_name"],
                "layers":     cfg["layers"],
            }

    # ----------------------------------------------------------------
    # Run one forward pass per unique combo, cache sequences by layer_key
    # ----------------------------------------------------------------
    # sequences_cache[layer_key] = {layer_index: (T, hidden_size) array}
    sequences_cache = {}

    # Cache of loaded wav2vec2 models to avoid reloading same model twice
    wav2vec2_model_cache = {}

    for layer_key, combo in unique_combos.items():
        model_name = combo["model_name"]
        layers     = combo["layers"]

        # Load or reuse processor/model for this model_name
        if model_name not in wav2vec2_model_cache:
            if processor is not None and wav2vec2_model is not None and model_name == gv.WAV2VEC2_MODEL_NAME:
                # Use pre-loaded model if it matches the default model name
                wav2vec2_model_cache[model_name] = (processor, wav2vec2_model)
            else:
                print(f"  Loading wav2vec2: {model_name}")
                proc, mdl = load_wav2vec2(model_name)
                wav2vec2_model_cache[model_name] = (proc, mdl)

        proc, mdl = wav2vec2_model_cache[model_name]

        # One forward pass extracts all needed layers at once
        sequences_cache[layer_key] = extract_wav2vec2_sequences_multilayer(
            audio, proc, mdl, layers
        )

    # ----------------------------------------------------------------
    # Scan each word in order
    # ----------------------------------------------------------------
    results       = []
    cursor_frame  = 0
    lookahead_frames = round(DEFAULT_LOOKAHEAD_S / FRAME_STRIDE_S)

    for word in words_in_order:
        cfg       = word_config[word]
        layer_key = cfg["layer_key"]

        # sequences_by_layer for this word: {layer_index: (T, hidden_size)}
        sequences_by_layer = sequences_cache[layer_key]

        model_path   = get_model_path_fn(word)
        metadata_csv = get_metadata_csv_fn(word)
        features_csv = get_features_csv_fn(word)

        expected_duration = get_word_expected_duration(word, metadata_csv)
        phonetic_fallback = get_word_typical_phonetic_score(word, features_csv)

        window_frames_options = [
            round((expected_duration * m) / FRAME_STRIDE_S)
            for m in WINDOW_SIZE_MULTIPLIERS
        ]

        scan_result = scan_word_in_sequence(
            word=word,
            sequences_by_layer=sequences_by_layer,
            model_path=model_path,
            cursor_frame=cursor_frame,
            max_lookahead_frames=lookahead_frames,
            window_frames_options=window_frames_options,
            phonetic_score_fallback=phonetic_fallback,
        )

        if scan_result is None:
            # Word not found with sufficient confidence anywhere in range —
            # flag for manual review. Cursor advances by expected duration
            # to keep pace with real speaking rate.
            results.append({
                "subject_id":         subject_id,
                "target_word":        word,
                "final_decision":     "no_match_found",
                "decision_reason":    "no_window_cleared_confidence_threshold",
                "probability_correct": None,
                "oneclass_score":     None,
                "window_start_s":     None,
                "window_end_s":       None,
                "review_required":    True,
            })
            cursor_frame += round(expected_duration / FRAME_STRIDE_S)
            continue

        # Apply decision rules — no phonetic_score or transcription available
        # in continuous scan mode, so those are passed as None.
        # The OC gate still fires if the word is in ONECLASS_GATE_PARAMS
        # and both conditions (probability + OC score) are met.
        decision = make_final_decision(
            target_word=word,
            probability_correct=scan_result["probability_correct"],
            phonetic_score=None,
            probability_threshold=DEFAULT_PROBABILITY_THRESHOLD,
            phonetic_cutoff=None,
            matched_transcription=None,
            detected_status=None,
            whisper_similarity=None,
            whisper_transcription=None,
            oneclass_score=scan_result["oneclass_score"],
        )

        decision["subject_id"]         = subject_id
        decision["target_word"]        = word
        decision["probability_correct"] = scan_result["probability_correct"]
        decision["oneclass_score"]     = scan_result["oneclass_score"]
        decision["window_start_s"]     = scan_result["window_start_frame"] * FRAME_STRIDE_S
        decision["window_end_s"]       = scan_result["window_end_frame"] * FRAME_STRIDE_S

        results.append(decision)
        cursor_frame = scan_result["window_end_frame"]

    return results


# ============================================================
# CLI ENTRY POINT
# ============================================================

def main():
    """
    Simple CLI for standalone testing, independent of predict.py's CLI.

    Example:
        python continuous_scan.py --audio path/to/full_recording.wav --subject-id 1973
    """

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--audio",      required=True,  help="Path to continuous recording .wav")
    parser.add_argument("--subject-id", required=True,  help="Subject ID")
    args = parser.parse_args()

    # Read target words in order from text file
    with open(gv.folder1 + f'texts/{gv.text}_1920/alltext.txt', encoding="utf-8") as f:
        words_in_order = f.read().split()
    words_in_order = words_in_order[:24]

    def get_model_path(word):
        """Resolve model.joblib path using per-word model and layer config."""
        model_name      = WAV2VEC2_MODEL_PER_WORD.get(word, gv.WAV2VEC2_MODEL_NAME)
        safe_model_name = model_name.replace("/", "__")
        if word in EMBEDDING_LAYERS_COMBINE_PER_WORD:
            layer_folder = "layer_combined"
        else:
            layer        = gv.EMBEDDING_LAYER_PER_WORD.get(word, gv.EMBEDDING_LAYER)
            layer_folder = f"layer_{layer}"
        safe_word = word.replace(" ", "_")
        return os.path.join("outputs", safe_model_name, layer_folder, safe_word, "model.joblib")

    def get_metadata_csv(word):
        safe_word = word.replace(" ", "_")
        return os.path.join("data", safe_word, "metadata.csv")

    def get_features_csv(word):
        """Resolve features.csv path using per-word model and layer config."""
        model_name      = WAV2VEC2_MODEL_PER_WORD.get(word, gv.WAV2VEC2_MODEL_NAME)
        safe_model_name = model_name.replace("/", "__")
        if word in EMBEDDING_LAYERS_COMBINE_PER_WORD:
            layer_folder = "layer_combined"
        else:
            layer        = gv.EMBEDDING_LAYER_PER_WORD.get(word, gv.EMBEDDING_LAYER)
            layer_folder = f"layer_{layer}"
        safe_word = word.replace(" ", "_")
        return os.path.join("outputs", safe_model_name, layer_folder, safe_word, "features.csv")

    results = predict_continuous_audio(
        audio_path=args.audio,
        words_in_order=words_in_order,
        get_model_path_fn=get_model_path,
        get_metadata_csv_fn=get_metadata_csv,
        get_features_csv_fn=get_features_csv,
        subject_id=args.subject_id,
    )

    df = pd.DataFrame(results)
    output_path = f"scan_predictions_{args.subject_id}.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {output_path}")
    print(df[["target_word", "final_decision", "probability_correct", "window_start_s", "window_end_s"]])


if __name__ == "__main__":
    main()