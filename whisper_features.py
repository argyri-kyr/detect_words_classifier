
import whisper


from globalVars import WHISPER_MODEL_SIZE
from load_phonetic_scores import normalize_greek_phonetic


def load_whisper_model():
    """
    Load the Whisper model for free transcription.

    We use large-v2 because it handles Greek child speech best —
    robust to slow reading, pronunciation variation, and accent
    differences across age groups and regions.

    Why free transcription (no forced alignment):
        The forced aligner maps audio to the target word even when
        the child said something different — giving a falsely high
        phonetic score. Free Whisper hears what was actually said,
        providing an honest phoneme-level signal.
    """

    print(f"Loading Whisper model ({WHISPER_MODEL_SIZE})...")
    whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
    return whisper_model


def get_whisper_similarity(audio_path, target_word, whisper_model):
    """
    Freely transcribe the audio with Whisper and compute phonetic
    similarity to the target word.

    Why this catches forced-aligner errors:
        The forced aligner gave phonetic_score=100 to αδευτικός
        because it mapped it to αρδευτικός. Free Whisper transcribes
        αδευτικός as-is, giving a low similarity score and correctly
        flagging the mispronunciation.

    No initial_prompt is used — this is intentional. We want Whisper
    to hear the audio freely, not be guided toward the target word.

    Returns:
        similarity: float 0-100
            100 = Whisper heard exactly the target word (phonetically)
            0   = Whisper heard something completely different
    """

    from rapidfuzz import fuzz

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