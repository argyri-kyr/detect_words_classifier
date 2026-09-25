

import samplerate  # noqa: F401 — forces eager import, avoids lazy_loader bug
import os
import numpy as np
import librosa

from globalVars import SAMPLE_RATE


def load_and_preprocess_audio(audio_path):
    """
    Load and preprocess one audio file.

    Steps:
        1. Load as mono at SAMPLE_RATE
        2. Trim silence from beginning and end
        3. Normalize volume to peak=1.0

    Why normalize volume:
        Reduces differences caused only by microphone loudness,
        not by pronunciation quality.

    Returns:
        audio: normalized 1D numpy array
    """

    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    audio, _ = librosa.effects.trim(audio, top_db=30)

    if len(audio) == 0:
        raise ValueError(f"Audio is empty after trimming: {audio_path}")

    max_value = np.max(np.abs(audio))
    if max_value > 0:
        audio = audio / max_value

    return audio


def extract_subject_id_from_path(audio_path):
    """
    Extract subject id from filename.

    Example:
        /path/αρδευτικός_correct/13784_αρδευτικός.wav
        → "13784"

    Works for all words as long as filenames follow:
        subjectid_word.wav
    """

    filename = os.path.basename(audio_path)
    subject_id = filename.split("_")[0]
    return subject_id