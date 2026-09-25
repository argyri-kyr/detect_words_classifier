import os
import re
import unicodedata
import pandas as pd

from globalVars import folder1, folder2, folder3

# ---------------------------------------------------------------------------
# Loads the phonetic score for a given subject and word from the
# detected words Excel file: *_detected_words_v3_{sub_id}.xlsx
#
# The phonetic_score in that file is computed by find_word_boundaries()
# using fuzz.ratio on normalized Greek phonetics. It handles:
#   - syllabic reading attempts (παυσι + πονας → παυσίπονο)
#   - merging of split detections
#   - look-back / look-ahead for reading attempts
#   - mispronunciations (αρδευτικό → αρδευτικός)
#
# This is a richer signal than the raw wav2vec2 alignment confidence score.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Greek phonetic normalization — must match detect_words.py exactly
# so that word matching is consistent between pipelines
# ---------------------------------------------------------------------------

def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFD', input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])


def normalize_greek_phonetic(text):
    if not text:
        return ""

    text = text.lower().strip()

    # αϊ/εϊ must be protected BEFORE remove_accents(), otherwise the
    # diaeresis is lost and αϊ becomes αι, wrongly matching the
    # αι -> ε diphthong rule below.
    text = text.replace('αϊ', 'X1').replace('εϊ', 'X2').replace('οϊ', 'X3').replace('υϊ', 'X4')

    text = remove_accents(text)
    
    # ξ = κσ phonetically in Greek — the forced aligner sometimes
    # transcribes εκσυγχρονισμένος as εξυγχρονισμένος.
    # Normalising ξ → κσ before all other rules ensures both forms
    # compare equal during transcription matching and phonetic scoring.
    text = text.replace('ξ', 'κσ')

    text = text.replace('αυ', 'αβ')
    text = text.replace('ευ', 'εβ')
    text = text.replace('ηυ', 'η')
    text = text.replace('αι', 'ε')
    text = text.replace('ου', 'U_PLACE')
    text = re.sub(r'ει|οι|υι', 'η', text)
    text = text.replace('αι', 'ε')
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )
    text = text.replace('υ', 'η')
    text = text.replace('ω', 'ο')
    text = text.replace('η', 'ι')
    text = text.replace('U_PLACE', 'ου')

    # Restore αϊ/εϊ as separate vowels (final form: αι / ει).
    text = text.replace('X1', 'αη').replace('X2', 'εη').replace('X3', 'οη').replace('X4', 'υη')

    return text


# ---------------------------------------------------------------------------
# File search helpers
# ---------------------------------------------------------------------------

def find_subject_json(sub_id):
    # Search through all folder3 candidates to find where subject_{sub_id} lives
    for f3 in folder3:
        subject_dir = os.path.join(folder1, f3, f"subject_{sub_id}")
        if not os.path.exists(subject_dir):
            continue
        # Filename prefix varies per folder (decodewordsv2 vs decodewordspseudo)
        for filename in os.listdir(subject_dir):
            if filename.endswith(f"_raw_detected_words_{sub_id}.json"):
                return os.path.join(subject_dir, filename)

    searched = [os.path.join(folder1, f3, f"subject_{sub_id}") for f3 in folder3]
    raise FileNotFoundError(
        f"❌ JSON not found for subject {sub_id} in any folder3 path.\n"
        f"   Searched:\n" + "\n".join(f"     {p}" for p in searched)
    )


def find_subject_excel(sub_id):
    # Build list of all paths to search — folder3 list + new_subjects
    search_paths = [
        os.path.join(folder1, f3, f"subject_{sub_id}")
        for f3 in folder3
    ]

    # Also search new_subjects folder
    new_subjects_path = os.path.join(folder1, folder2, "new_subjects", f"subject_{sub_id}")
    if new_subjects_path not in search_paths:
        search_paths.append(new_subjects_path)

    for subject_dir in search_paths:
        if not os.path.exists(subject_dir):
            continue
        for filename in os.listdir(subject_dir):
            if filename.endswith(f"_detected_words_v3_{sub_id}.xlsx"):
                return os.path.join(subject_dir, filename)

    raise FileNotFoundError(
        f"❌ Excel not found for subject {sub_id} in any path.\n"
        f"   Searched:\n" + "\n".join(f"     {p}" for p in search_paths)
    )


def find_matching_word_row(df, word):
    """
    Finds the row in the detected-words Excel file that corresponds
    to the target word.

    First it tries exact word match.
    If that fails, it uses Greek phonetic normalization.
    """

    if "word" not in df.columns:
        raise ValueError(
            f"❌ Excel is missing required column: word\n"
            f"   Columns found: {list(df.columns)}"
        )

    target_norm = normalize_greek_phonetic(word)

    # First try exact match.
    exact = df[df["word"] == word]
    if not exact.empty:
        return exact.iloc[0]

    # Then try normalized phonetic match.
    df = df.copy()
    df["word_norm"] = df["word"].apply(
        lambda w: normalize_greek_phonetic(str(w)) if pd.notna(w) else ""
    )

    norm_match = df[df["word_norm"] == target_norm]
    if not norm_match.empty:
        return norm_match.iloc[0]

    raise ValueError(
        f"❌ Word '{word}' was not found in detected-words Excel.\n"
        f"   Normalized target: {target_norm}\n"
        f"   Words found: {list(df['word'].dropna())}"
    )


def load_detected_word_info(sub_id, word):
    """
    Loads useful information from:
        *_detected_words_v3_{sub_id}.xlsx

    We intentionally do NOT return start_s or end_s because they are not
    useful for the current classifier/reporting logic.

    Returned fields:
        phonetic_score
        matched_transcription
        detected_status
    """

    excel_path = find_subject_excel(sub_id)
    df = pd.read_excel(excel_path)

    # Handle both old column name (matched_transcription) and
    # new column names (matched_transcription_raw/norm/display)
    if "matched_transcription" not in df.columns:
        if "matched_transcription_display" in df.columns:
            df["matched_transcription"] = df["matched_transcription_display"]
        elif "matched_transcription_norm" in df.columns:
            df["matched_transcription"] = df["matched_transcription_norm"]
        elif "matched_transcription_raw" in df.columns:
            df["matched_transcription"] = df["matched_transcription_raw"]

    required_columns = ["word", "phonetic_score", "matched_transcription", "status"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"❌ Excel for subject {sub_id} is missing required columns: {missing_columns}\n"
            f"   File: {excel_path}\n"
            f"   Columns found: {list(df.columns)}"
        )

    row = find_matching_word_row(df, word)

    return {
        "phonetic_score": float(row["phonetic_score"]),
        "matched_transcription": row["matched_transcription"],
        "detected_status": row["status"],
    }
    
    
# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def load_phonetic_score(sub_id, word):
    """
    Backwards-compatible function.

    Existing code can still call load_phonetic_score().
    Internally, it now uses load_detected_word_info().
    """

    info = load_detected_word_info(sub_id, word)
    return info["phonetic_score"]