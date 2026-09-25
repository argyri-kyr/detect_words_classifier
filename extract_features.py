# extract_features.py

import os
import numpy as np
import pandas as pd
from tqdm import tqdm


from globalVars import (
    LABEL_TO_ID, USE_PHONETIC_SCORE, USE_WHISPER_SIMILARITY,
    USE_MEL_PROTOTYPE, EMBEDDING_LAYERS_COMBINE_PER_WORD
)
from wav2vec2_features import (
    load_wav2vec2, extract_wav2vec2_embedding,
    extract_wav2vec2_embedding_multilayer
)

from utils import read_metadata, save_csv
from load_phonetic_scores import load_detected_word_info

from audio_utils import load_and_preprocess_audio, extract_subject_id_from_path
from wav2vec2_features import load_wav2vec2, extract_wav2vec2_embedding
from whisper_features import load_whisper_model, get_whisper_similarity
from mel_prototype import compute_mel_spectrogram, compute_medoid, compute_dtw_distance_to_medoid


def create_features_csv(metadata_csv, output_csv, model_name):
    """
    Creates the features CSV using a two-pass extraction pipeline.

    Pass 1 — for each audio file:
        - Load and preprocess audio (once, reused for all features)
        - Extract wav2vec2 embedding
        - Load phonetic score from Excel  (if USE_PHONETIC_SCORE)
        - Compute Whisper similarity      (if USE_WHISPER_SIMILARITY)
        - Compute mel spectrogram         (if USE_MEL_PROTOTYPE, stored for pass 2)

    Between passes:
        - Find all correct samples
        - Compute their mel spectrogram medoid (most central correct pronunciation)

    Pass 2 — for each audio file:
        - Compute DTW distance to medoid  (if USE_MEL_PROTOTYPE)
        - Assemble final f_* feature vector
        - Build output row

    Feature order (must match predict.py exactly):
        f_0 ... f_N   wav2vec2 mean+std embedding
        f_N+1         phonetic_score          (if USE_PHONETIC_SCORE)
        f_N+2         whisper_similarity      (if USE_WHISPER_SIMILARITY)
        f_N+3         dtw_distance_to_medoid  (if USE_MEL_PROTOTYPE)

    Diagnostic-only columns saved but NOT in f_* features:
        matched_transcription, detected_status, whisper_similarity (raw)

    Returns:
        medoid_sequence: the correct pronunciation medoid (numpy array)
                         saved into model.joblib so predict.py can use it.
                         None if USE_MEL_PROTOTYPE is False.
    """

    metadata = read_metadata(metadata_csv)

    processor, wav2vec2_mod = load_wav2vec2(model_name)

    whisper_model = None
    if USE_WHISPER_SIMILARITY:
        whisper_model = load_whisper_model()

    # ============================================================
    # PASS 1
    # ============================================================
    print("Pass 1: extracting embeddings and features...")

    pass1_rows = []
    mel_sequences = []

    for _, row in tqdm(metadata.iterrows(), total=len(metadata)):

        audio_path = row["filepath"]
        target_word = row["target_word"]
        label = row["label"]
        filename = os.path.basename(audio_path)
        label_id = LABEL_TO_ID[label]

        # Load audio once — reused for wav2vec2, mel, and Whisper.
        audio = load_and_preprocess_audio(audio_path)

        # Feature 1: wav2vec2 embedding.
        if target_word in EMBEDDING_LAYERS_COMBINE_PER_WORD:
            layers = EMBEDDING_LAYERS_COMBINE_PER_WORD[target_word]
            embedding = extract_wav2vec2_embedding_multilayer(
                audio, processor, wav2vec2_mod, layers
            )
        else:
            embedding = extract_wav2vec2_embedding(audio, processor, wav2vec2_mod)

        # Feature 2: phonetic score from Excel.
        subject_id = extract_subject_id_from_path(audio_path)
        phonetic_score = None
        matched_transcription = None
        detected_status = None

        if USE_PHONETIC_SCORE:
            detected_info = load_detected_word_info(
                sub_id=subject_id,
                word=target_word,
            )
            phonetic_score = detected_info["phonetic_score"]
            matched_transcription = detected_info["matched_transcription"]
            detected_status = detected_info["detected_status"]
            embedding = np.append(embedding, phonetic_score)

        # Feature 3: Whisper free transcription similarity.
        whisper_similarity = None
        whisper_transcription = None
        if USE_WHISPER_SIMILARITY:
            whisper_similarity, whisper_transcription = get_whisper_similarity(
                audio_path=audio_path,
                target_word=target_word,
                whisper_model=whisper_model,
            )
            embedding = np.append(embedding, whisper_similarity)

        # Mel spectrogram — stored temporarily for pass 2.
        mel_seq = None
        if USE_MEL_PROTOTYPE:
            mel_seq = compute_mel_spectrogram(audio)

        mel_sequences.append(mel_seq)

        pass1_rows.append({
            "audio_path": audio_path,
            "filename": filename,
            "word": target_word,
            "label": label,
            "label_id": label_id,
            "phonetic_score": phonetic_score,
            "whisper_similarity": whisper_similarity,
            "whisper_transcription": whisper_transcription,
            "matched_transcription": matched_transcription,
            "detected_status": detected_status,
            "embedding": embedding,
        })

    # ============================================================
    # Between passes — compute correct pronunciation medoid.
    # ============================================================
    medoid_sequence = None

    if USE_MEL_PROTOTYPE:
        print("Computing correct pronunciation medoid...")

        correct_sequences = [
            mel_sequences[i]
            for i, r in enumerate(pass1_rows)
            if r["label"] == "correct"
        ]

        if len(correct_sequences) == 0:
            raise ValueError("No correct samples found — cannot compute medoid.")

        medoid_sequence, _ = compute_medoid(correct_sequences)
        print(f"✅ Medoid computed from {len(correct_sequences)} correct samples.")

    # ============================================================
    # PASS 2
    # ============================================================
    print("Pass 2: computing DTW distances and building feature rows...")

    rows = []

    for i, r in enumerate(pass1_rows):
        embedding = r["embedding"]

        # Feature 4: DTW distance to correct medoid.
        if USE_MEL_PROTOTYPE and medoid_sequence is not None:
            dtw_dist = compute_dtw_distance_to_medoid(
                mel_sequence=mel_sequences[i],
                medoid_sequence=medoid_sequence,
            )
            embedding = np.append(embedding, dtw_dist)

        feature_values = {
            f"f_{j}": value
            for j, value in enumerate(embedding)
        }

        output_row = {
            "audio_path": r["audio_path"],
            "filename": r["filename"],
            "word": r["word"],
            "label": r["label"],
            "label_id": r["label_id"],
            "phonetic_score": r["phonetic_score"],
            "whisper_similarity": r["whisper_similarity"],
            "whisper_transcription": r["whisper_transcription"],
            "matched_transcription": r["matched_transcription"],
            "detected_status": r["detected_status"],
        }

        output_row.update(feature_values)
        rows.append(output_row)

    features_df = pd.DataFrame(rows)
    save_csv(features_df, output_csv)

    print(f"✅ Saved features to: {output_csv}")

    return medoid_sequence