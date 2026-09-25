# predict.py
#
# Predicts the pronunciation quality of a single audio clip for one word.
#
# Entry points:
#   predict_one_audio() — called programmatically by predict_new_subjects_main.py
#   main()              — called from command line for single-file testing
#
# Output (final_decision) can be:
#   correct            = automatically accepted as correct pronunciation
#   incorrect          = rejected as wrong pronunciation
#   possible_correct   = send to teacher (probably correct)
#   possible_incorrect = send to teacher (probably incorrect)
#
# Pipeline:
#   1. Load model.joblib (binary SVC Pipeline: PCA + scaler + SVC)
#   2. Load model_oneclass.joblib (OC-SVM gate, if it exists)
#   3. Load and preprocess audio
#   4. Extract features matching those used during training
#   5. Compute OC-SVM score
#   6. Run binary SVC → probability_correct
#   7. Apply decision rules → final_decision

import argparse
import joblib
import numpy as np
import pandas as pd
import os

import globalVars as gv

from audio_utils import load_and_preprocess_audio, extract_subject_id_from_path
from wav2vec2_features import (
    load_wav2vec2,
    extract_wav2vec2_embedding,
    extract_wav2vec2_embedding_multilayer,
)
from whisper_features import load_whisper_model, get_whisper_similarity
from mel_prototype import compute_mel_spectrogram, compute_dtw_distance_to_medoid

from load_phonetic_scores import load_detected_word_info
from decision_rules import make_final_decision
from globalVars import (
    USE_WHISPER_SIMILARITY,
    USE_MEL_PROTOTYPE,
    EMBEDDING_LAYERS_COMBINE_PER_WORD,
)


def predict_one_audio(
    model_path,
    audio_path,
    target_word,
    subject_id=None,
    processor=None,
    wav2vec2_model=None,
):
    """
    Predicts pronunciation quality for one audio clip of one word.

    Feature extraction must exactly match what create_features_csv()
    did during training — same model, same layers, same feature order.
    The trained model.joblib stores everything needed to reproduce
    the training conditions at inference time.

    Feature order (must match extract_features.py):
        1. wav2vec2 mean+std embedding (single or multi-layer)
        2. phonetic_score              (if USE_PHONETIC_SCORE)
        3. whisper_similarity          (if USE_WHISPER_SIMILARITY)
        4. dtw_distance_to_medoid      (if USE_MEL_PROTOTYPE)

    PCA note:
        PCA is part of the sklearn Pipeline saved inside model.joblib.
        It is applied automatically by classifier.predict_proba() —
        no explicit PCA step is needed here.

    Args:
        model_path:     path to model.joblib for this word
        audio_path:     path to the .wav audio clip to classify
        target_word:    the Greek word being assessed (e.g. "αρδευτικός")
        subject_id:     child's ID; if None, extracted from filename
        processor:      pre-loaded wav2vec2 processor (optional, for speed)
        wav2vec2_model: pre-loaded wav2vec2 model (optional, for speed)

    Returns:
        dict with final_decision, probability_correct, phonetic_score,
        oneclass_score, decision_reason, and all input metadata.
    """

    # ----------------------------------------------------------------
    # Step 1: Load trained binary SVC and its saved training parameters
    # ----------------------------------------------------------------
    model_data = joblib.load(model_path)

    classifier           = model_data["model"]         # sklearn Pipeline (PCA+scaler+SVC)
    feature_columns      = model_data["feature_columns"]  # list of f_0..f_N column names
    best_threshold       = model_data.get("best_threshold", 0.50)       # CV-selected probability cutoff
    best_phonetic_cutoff = model_data.get("best_phonetic_cutoff", None) # CV-selected phonetic cutoff
    trained_layer        = model_data.get("embedding_layer", gv.EMBEDDING_LAYER) # single-layer index
    medoid_sequence      = model_data.get("medoid_sequence", None)      # mel medoid for DTW
    trained_model_name   = model_data.get("model_name", gv.WAV2VEC2_MODEL_NAME)  # wav2vec2 model name

    # ----------------------------------------------------------------
    # Step 2: Load one-class SVM gate if it exists for this word
    # The gate provides a second opinion alongside the binary SVC.
    # It scores how closely the audio resembles correct pronunciations.
    # ----------------------------------------------------------------
    oneclass_score         = None
    ocsvm_pred_model       = None
    ocsvm_pred_feature_cols = None

    ocsvm_path = os.path.normpath(
        os.path.join(os.path.dirname(model_path), "model_oneclass.joblib")
    )

    if os.path.exists(ocsvm_path):
        ocsvm_data             = joblib.load(ocsvm_path)
        ocsvm_pred_model       = ocsvm_data["model"]
        ocsvm_pred_feature_cols = ocsvm_data["feature_columns"]

    # ----------------------------------------------------------------
    # Step 3: Load wav2vec2 model
    # Uses pre-loaded model if provided (faster for batch processing).
    # Otherwise loads the model name saved at training time — this
    # ensures we use the same model that produced the training features.
    # ----------------------------------------------------------------
    if processor is None or wav2vec2_model is None:
        processor, wav2vec2_model = load_wav2vec2(trained_model_name)

    # ----------------------------------------------------------------
    # Step 4a: Load and preprocess audio
    # Audio is loaded once and reused for all feature extractors.
    # Preprocessing: mono, 16kHz, silence trim, peak normalisation.
    # ----------------------------------------------------------------
    audio = load_and_preprocess_audio(audio_path)

    # ----------------------------------------------------------------
    # Step 4b: Extract wav2vec2 embedding — Feature 1
    #
    # Two paths depending on whether this word uses combined layers:
    #
    # Combined-layer path (word in EMBEDDING_LAYERS_COMBINE_PER_WORD):
    #   Extracts embeddings from multiple layers (e.g. [6, 9, 12]) and
    #   concatenates them. The layer list is read from globalVars, not
    #   from the model file, so globalVars must match training settings.
    #   No gv.EMBEDDING_LAYER override needed — layers are explicit.
    #
    # Single-layer path (all other words):
    #   Temporarily overrides gv.EMBEDDING_LAYER to trained_layer so
    #   extract_wav2vec2_embedding() uses the correct layer.
    #   The override is restored in the finally block even if an error
    #   occurs, so other words are not affected.
    # ----------------------------------------------------------------
    original_layer = gv.EMBEDDING_LAYER

    try:
        if target_word in EMBEDDING_LAYERS_COMBINE_PER_WORD:
            # Multi-layer: concatenate embeddings from all listed layers
            layers   = EMBEDDING_LAYERS_COMBINE_PER_WORD[target_word]
            features = extract_wav2vec2_embedding_multilayer(
                audio, processor, wav2vec2_model, layers
            )
        else:
            # Single-layer: temporarily set the trained layer
            gv.EMBEDDING_LAYER = trained_layer
            features = extract_wav2vec2_embedding(audio, processor, wav2vec2_model)
    finally:
        # Always restore the global layer, even if extraction failed
        gv.EMBEDDING_LAYER = original_layer

    # ----------------------------------------------------------------
    # Step 4c: Get subject ID for phonetic score lookup
    # ----------------------------------------------------------------
    if subject_id is None:
        subject_id = extract_subject_id_from_path(audio_path)

    # ----------------------------------------------------------------
    # Step 4d: Load phonetic score from Excel — Feature 2
    # Reads the forced-aligner output for this subject+word pair.
    # Returns: phonetic_score, matched_transcription, detected_status.
    # phonetic_score is appended to the feature vector if USE_PHONETIC_SCORE.
    # ----------------------------------------------------------------
    detected_info = load_detected_word_info(sub_id=subject_id, word=target_word)

    phonetic_score        = detected_info["phonetic_score"]
    matched_transcription = detected_info["matched_transcription"]
    detected_status       = detected_info["detected_status"]

    if gv.USE_PHONETIC_SCORE:
        features = np.append(features, phonetic_score)

    # ----------------------------------------------------------------
    # Step 4e: Whisper similarity — Feature 3 (currently disabled)
    # When enabled, freely transcribes the audio with Whisper and
    # computes phonetic similarity to the target word. Disabled because
    # it hurts accuracy on child speech (USE_WHISPER_SIMILARITY=False).
    # ----------------------------------------------------------------
    whisper_similarity    = None
    whisper_transcription = None
    if USE_WHISPER_SIMILARITY:
        whisper_model_obj = load_whisper_model()
        whisper_similarity, whisper_transcription = get_whisper_similarity(
            audio_path=audio_path,
            target_word=target_word,
            whisper_model=whisper_model_obj,
        )
        features = np.append(features, whisper_similarity)

    # ----------------------------------------------------------------
    # Step 4f: DTW distance to mel-spectrogram medoid — Feature 4
    # (currently disabled, USE_MEL_PROTOTYPE=False due to leakage fix
    # not yet implemented). When enabled, measures how different the
    # audio's mel-spectrogram is from the medoid of correct samples.
    # ----------------------------------------------------------------
    dtw_distance = None
    if USE_MEL_PROTOTYPE and medoid_sequence is not None:
        mel_seq      = compute_mel_spectrogram(audio)
        dtw_distance = compute_dtw_distance_to_medoid(mel_seq, medoid_sequence)
        features     = np.append(features, dtw_distance)

    # ----------------------------------------------------------------
    # Step 4g: Feature length sanity check
    # If this fails, the feature flags (USE_PHONETIC_SCORE etc.) or
    # layer settings at prediction time don't match training time.
    # ----------------------------------------------------------------
    if len(features) != len(feature_columns):
        raise ValueError(
            "Feature length mismatch — prediction settings don't match training.\n"
            f"  Extracted features:    {len(features)}\n"
            f"  Model feature columns: {len(feature_columns)}\n"
            "  Check USE_PHONETIC_SCORE, USE_WHISPER_SIMILARITY, "
            "USE_MEL_PROTOTYPE, and EMBEDDING_LAYERS_COMBINE_PER_WORD."
        )

    # ----------------------------------------------------------------
    # Step 5: Compute one-class SVM gate score
    # The OC-SVM operates on the raw feature vector (before PCA) using
    # its own internal StandardScaler. A negative score means the audio
    # is outside the learned correct-pronunciation boundary.
    # This score is later used by make_final_decision() via ONECLASS_GATE_PARAMS.
    # ----------------------------------------------------------------
    if ocsvm_pred_model is not None:
        try:
            row_for_oc = {col: val for col, val in zip(feature_columns, features)}
            X_oc       = pd.DataFrame([row_for_oc])[ocsvm_pred_feature_cols]
            oneclass_score = float(ocsvm_pred_model.decision_function(X_oc)[0])
        except Exception:
            oneclass_score = None

    # ----------------------------------------------------------------
    # Step 6: Binary SVC classification
    # The Pipeline applies PCA + StandardScaler internally before SVC.
    # predict_proba returns [prob_incorrect, prob_correct]; we take [:,1].
    # ----------------------------------------------------------------
    row = {col: val for col, val in zip(feature_columns, features)}
    X   = pd.DataFrame([row])
    probability_correct = float(classifier.predict_proba(X)[0][1])

    # ----------------------------------------------------------------
    # Step 7: Apply decision rules → final_decision
    # Combines probability_correct, phonetic_score, transcription match,
    # and oneclass_score into one of four possible outputs.
    # See decision_rules.py for the full rule set.
    # ----------------------------------------------------------------
    decision = make_final_decision(
        target_word=target_word,
        probability_correct=probability_correct,
        phonetic_score=phonetic_score,
        probability_threshold=best_threshold,
        phonetic_cutoff=best_phonetic_cutoff,
        matched_transcription=matched_transcription,
        detected_status=detected_status,
        whisper_similarity=whisper_similarity,
        whisper_transcription=whisper_transcription,
        oneclass_score=oneclass_score,
    )

    # Attach all metadata for logging and debugging
    decision["audio_path"]             = audio_path
    decision["subject_id"]             = subject_id
    decision["target_word"]            = target_word
    decision["embedding_layer"]        = trained_layer
    decision["wav2vec2_model"]         = trained_model_name
    decision["probability_correct"]    = probability_correct
    decision["probability_threshold"]  = best_threshold
    decision["phonetic_score"]         = phonetic_score
    decision["phonetic_cutoff"]        = best_phonetic_cutoff
    decision["whisper_similarity"]     = whisper_similarity
    decision["whisper_transcription"]  = whisper_transcription
    decision["dtw_distance_to_medoid"] = dtw_distance
    decision["matched_transcription"]  = matched_transcription
    decision["detected_status"]        = detected_status
    decision["oneclass_score"]         = oneclass_score

    return decision


def main():
    """
    Command-line entry point for single-file prediction.

    Usage:
        python predict.py --model outputs/.../model.joblib
                          --audio /path/to/clip.wav
                          --word  αρδευτικός
                          [--subject-id 12345]
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      required=True,  help="Path to model.joblib")
    parser.add_argument("--audio",      required=True,  help="Path to audio .wav file")
    parser.add_argument("--word",       required=True,  help="Target word, e.g. αρδευτικός")
    parser.add_argument("--subject-id", required=False, default=None,
                        help="Subject ID (extracted from filename if omitted)")
    args = parser.parse_args()

    result = predict_one_audio(
        model_path=args.model,
        audio_path=args.audio,
        target_word=args.word,
        subject_id=args.subject_id,
    )

    print("\n============================================================")
    print("  PREDICTION")
    print("============================================================\n")

    print(f"Audio:                   {result['audio_path']}")
    print(f"Subject id:              {result['subject_id']}")
    print(f"Target word:             {result['target_word']}")
    print(f"wav2vec2 model:          {result['wav2vec2_model']}")
    print(f"Embedding layer:         {result['embedding_layer']}")

    print(f"\nModel prediction:        {result['model_prediction']}")
    print(f"Final decision:          {result['final_decision']}")
    print(f"Decision reason:         {result['decision_reason']}")

    print(f"\nProbability correct:     {result['probability_correct']:.4f}")
    print(f"Probability threshold:   {result['probability_threshold']}")

    print(f"\nPhonetic score:          {result['phonetic_score']}")
    print(f"Phonetic cutoff:         {result['phonetic_cutoff']}")

    print(f"\nWhisper transcription:   {result.get('whisper_transcription', 'N/A')}")
    print(f"Whisper similarity:      {result['whisper_similarity']}")
    print(f"DTW distance to medoid:  {result['dtw_distance_to_medoid']}")

    print(f"\nMatched transcription:   {result['matched_transcription']}")
    print(f"Detected status:         {result['detected_status']}")
    print(f"Transcription match:     {result['transcription_exact_match']}")
    print(f"One-class score:         {result['oneclass_score']}")

    print("\nOutput meaning:")
    print("  correct            = automatically accepted")
    print("  incorrect          = rejected")
    print("  possible_correct   = listen manually, probably correct")
    print("  possible_incorrect = listen manually, probably incorrect")


if __name__ == "__main__":
    main()