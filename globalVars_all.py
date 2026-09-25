# globalVars.py
#
# Central configuration file for the Greek child speech pronunciation classifier.
# All pipeline settings live here — models, features, training, decision rules.
#
# Sections:
#   1. Data folders
#   2. Audio settings
#   3. Label mappings
#   4. General
#   5. wav2vec2 model settings
#   6. Feature flags
#   7. Whisper settings
#   8. SVC training settings
#   9. Train/test split settings
#  10. Data augmentation settings
#  11. Decision rule settings
#  12. One-class SVM gate parameters


# ============================================================
# 1. DATA FOLDERS
# ============================================================

# Root folder for all project data
folder1 = "/_Iro_analysis_tobii7_HD/"

# Subfolder containing the labeled word audio clips.
# Each word has a subfolder with two subdirectories:
#   {word}/{word}_correct/   <- correct pronunciation clips
#   {word}/{word}_wrong/     <- incorrect pronunciation clips
folder2 = 'detect_words_decodewordspseudo_data_all/'
# folder2 = 'detect_words_decodewordspseudo_data_clean/'
# folder2 = "detect_words_decodewordspseudo_data_raw/"
# folder2 = 'detect_words_decodewordspseudo_data_raw_renamed/'

# Folders searched when looking up per-subject Excel files
# (detected_words_v3_{sub_id}.xlsx) during feature extraction.
# Searched in order — first match wins.
folder3 = [
    "individual_decodewordsv2_Ierapetra",
    "individual_decodewordspseudo_larisa_cherso2",
    "individual_decodewordspseudo_apokor",
    "individual_decodewordspseudo_extra",
    "new_subjects",
]

# Text label used in filename conventions throughout the pipeline
text = "decodewordspseudo"


# ============================================================
# 2. AUDIO SETTINGS
# ============================================================

# Target sample rate for all audio loading and processing.
# wav2vec2 models expect 16kHz input.
SAMPLE_RATE = 16000

# Number of MFCC coefficients (kept for compatibility, not currently used
# in the main feature pipeline which uses wav2vec2 embeddings instead).
N_MFCC = 40


# ============================================================
# 3. LABEL MAPPINGS
# ============================================================

# Maps string labels to integer IDs for sklearn compatibility.
# "correct" = 1, "incorrect" = 0 throughout the pipeline.
LABEL_TO_ID = {
    "incorrect": 0,
    "correct":   1,
}

# Reverse mapping — used in evaluate.py to convert predictions back
# to human-readable labels for the output CSVs and reports.
ID_TO_LABEL = {
    0: "incorrect",
    1: "correct",
}


# ============================================================
# 4. GENERAL
# ============================================================

# Fixed random seed used everywhere for reproducibility:
# train/test splits, StratifiedKFold, SVC, PCA.
RANDOM_STATE = 42


# ============================================================
# 5. WAV2VEC2 MODEL SETTINGS
# ============================================================

# Default wav2vec2 model used for all words unless overridden below.
# lighteternal is a Greek-specific fine-tune of wav2vec2-large-xlsr-53
# and is the best general-purpose model for this task.
WAV2VEC2_MODEL_NAME = "lighteternal/wav2vec2-large-xlsr-53-greek"

# Per-word model selection.
# After comparing lighteternal vs skylord/wav2vec2-large-xlsr-greek-2
# across all 24 words, each word is assigned the model that gave the
# best combination of correct% and wrong_accepted=0.
# lighteternal wins on 14/24 words (safer, fewer wrong_accepted).
# skylord wins on 10/24 words (higher correct% or fewer correct_rejected).

WAV2VEC2_MODEL_PER_WORD = {
    # lighteternal — better correct% or safer (fewer wrong_accepted)
    "ακαδημαϊκά":       "lighteternal/wav2vec2-large-xlsr-53-greek",
    "αλάτια":           "lighteternal/wav2vec2-large-xlsr-53-greek",
    "αρδευτικός":       "lighteternal/wav2vec2-large-xlsr-53-greek",
    "αυθαιρεσίες":      "lighteternal/wav2vec2-large-xlsr-53-greek",
    "δεσμεύω":          "lighteternal/wav2vec2-large-xlsr-53-greek",
    "δραστηριότητες":   "lighteternal/wav2vec2-large-xlsr-53-greek",
    "εκσυγχρονισμένος": "lighteternal/wav2vec2-large-xlsr-53-greek",
    "επιχειρήματα":     "lighteternal/wav2vec2-large-xlsr-53-greek",
    "ηλεκτρική":        "lighteternal/wav2vec2-large-xlsr-53-greek",
    "λουλούδια":        "lighteternal/wav2vec2-large-xlsr-53-greek",
    "παυσίπονο":        "lighteternal/wav2vec2-large-xlsr-53-greek",
    "περιβάλλον":       "lighteternal/wav2vec2-large-xlsr-53-greek",
    "πρόσκληση":        "lighteternal/wav2vec2-large-xlsr-53-greek",
    "σκουπίδια":        "lighteternal/wav2vec2-large-xlsr-53-greek",
    # skylord — better correct% or fewer correct_rejected for these words
    "βοοειδές":         "skylord/wav2vec2-large-xlsr-greek-2",
    "εγκλείει":         "skylord/wav2vec2-large-xlsr-greek-2",
    "εκπαίδευση":       "skylord/wav2vec2-large-xlsr-greek-2",
    "μετεγγραφές":      "skylord/wav2vec2-large-xlsr-greek-2",
    "ποίημα":           "skylord/wav2vec2-large-xlsr-greek-2",
    "σταφιδιασμένα":    "skylord/wav2vec2-large-xlsr-greek-2",
    "συμπληρώσουμε":    "skylord/wav2vec2-large-xlsr-greek-2",
    "ταξιδεύουμε":      "skylord/wav2vec2-large-xlsr-greek-2",
    "τήβεννος":         "skylord/wav2vec2-large-xlsr-greek-2",
    "υπεύθυνος":        "skylord/wav2vec2-large-xlsr-greek-2",
}


# Default embedding layer when a word uses single-layer extraction.
# Middle layers (6-12) retain more acoustic/pronunciation detail than
# the final layer which is optimised for ASR word identity.
# Read at runtime via gv.EMBEDDING_LAYER — never import directly with
# "from globalVars import EMBEDDING_LAYER" as that copies the value
# at import time and misses the per-word override in main.py.
EMBEDDING_LAYER = 6

# Per-word embedding layer overrides for single-layer words.
# Only relevant for words NOT in EMBEDDING_LAYERS_COMBINE_PER_WORD.
# Currently only δεσμεύω uses single-layer extraction (layer 12).
EMBEDDING_LAYER_PER_WORD = {
    "δεσμεύω": 12,  # single layer — combined layers hurt this word
}

# Per-word layer combinations for multi-layer embedding extraction.
# For these words, embeddings from all listed layers are extracted
# and concatenated before PCA compression.
# Why [6, 9, 12]: layer 6 captures acoustic/phonetic detail,
# layer 9 captures intermediate representations, layer 12 captures
# higher-level linguistic content. Together they give the SVC a
# richer view than any single layer alone.
# δεσμεύω is intentionally excluded — single layer_12 works better.

EMBEDDING_LAYERS_COMBINE_PER_WORD = {
    "ακαδημαϊκά":       [6, 9, 12],
    "αλάτια":           [6, 9, 12],
    "αρδευτικός":       [6, 9, 12],
    "αυθαιρεσίες":      [6, 9, 12],
    "βοοειδές":         [6, 9, 12],
    "δραστηριότητες":   [6, 9, 12],
    "εγκλείει":         [6, 9, 12],
    "εκπαίδευση":       [6, 9, 12],
    "εκσυγχρονισμένος": [6, 9, 12],
    "επιχειρήματα":     [6, 9, 12],
    "ηλεκτρική":        [6, 9, 12],
    "λουλούδια":        [6, 9, 12],
    "μετεγγραφές":      [6, 9, 12],
    "παυσίπονο":        [6, 9, 12],
    "περιβάλλον":       [6, 9, 12],
    "ποίημα":           [6, 9, 12],
    "πρόσκληση":        [6, 9, 12],
    "σκουπίδια":        [6, 9, 12],
    "σταφιδιασμένα":    [6, 9, 12],
    "συμπληρώσουμε":    [6, 9, 12],
    "ταξιδεύουμε":      [6, 9, 12],
    "τήβεννος":         [6, 9, 12],
    "υπεύθυνος":        [6, 9, 12],
    # δεσμεύω NOT here — uses single layer_12 from EMBEDDING_LAYER_PER_WORD
}


# ============================================================
# 6. FEATURE FLAGS
# ============================================================

# Whether to append phonetic_score to the wav2vec2 embedding.
# The phonetic score comes from the forced aligner Excel file and
# measures how closely the child's pronunciation matched the target
# word at the phoneme level. Consistently improves classification.
USE_PHONETIC_SCORE = True

# Whether to compute Whisper free-transcription similarity as a feature.
# When enabled, Whisper transcribes the audio freely and the result is
# compared phonetically to the target word (fuzz.ratio, 0-100).
# DISABLED: tested and found to hurt accuracy on child speech — Whisper
# struggles with pseudowords and child pronunciation variations.
USE_WHISPER_SIMILARITY = False

# Whether to compute mel-spectrogram DTW distance to the correct
# pronunciation medoid as a feature.
# DISABLED: data leakage issue — the medoid is computed from ALL correct
# samples before train/test split, so test samples influence the feature.
# Can be re-enabled after refactoring medoid computation to happen inside
# train.py after the split.
USE_MEL_PROTOTYPE = False


# ============================================================
# 7. WHISPER SETTINGS
# ============================================================

# Whisper model size for free transcription (when USE_WHISPER_SIMILARITY=True).
# large-v3 is most accurate but use large-v2 if it causes OSError on Windows.
WHISPER_MODEL_SIZE = "large-v3"

# Minimum Whisper phonetic similarity score (0-100) to auto-accept a prediction.
# If Whisper heard something sufficiently different from the target word,
# the sample is sent to teacher review instead of auto-accepted.
# Currently unused since USE_WHISPER_SIMILARITY=False.
WHISPER_ACCEPT_THRESHOLD = 60.0


# ============================================================
# 8. SVC TRAINING SETTINGS
# ============================================================

# Grid of SVC configurations tested during grid search.
# For each config, cross-validation selects the best probability
# threshold and phonetic cutoff. The config with the best
# CV performance is used to train the final model.
# Tested alternatives (higher C, lower C + finer gamma) did not
# improve results — the original 5 configs are optimal for the
# PCA-compressed feature space.
SVC_GRID = [
    {"kernel": "rbf",    "C": 5,  "gamma": 0.001},
    {"kernel": "rbf",    "C": 10, "gamma": 0.001},
    {"kernel": "rbf",    "C": 1,  "gamma": 0.001},
    {"kernel": "rbf",    "C": 5,  "gamma": "scale"},
    {"kernel": "linear", "C": 1,  "gamma": "scale"},
]


# ============================================================
# 9. TRAIN/TEST SPLIT SETTINGS
# ============================================================

# Default fraction of data held out for the test set.
# 80% train / 20% test — standard split for most words.
TEST_SIZE_DEFAULT = 0.20

# Per-word overrides for small datasets.
# Words with very few total samples (~30) benefit from a smaller
# test set so the model has more training data. At 10%, βοοειδές
# gets ~3 test samples instead of 6 — still enough to evaluate,
# but gives the SVC 4 more training examples which matters significantly
# with small datasets and high-dimensional embeddings.
# Words not listed here use TEST_SIZE_DEFAULT.
TEST_SIZE_PER_WORD = {
    "βοοειδές": 0.10,
}


# ============================================================
# 10. DATA AUGMENTATION SETTINGS
# ============================================================

# Words whose incorrect samples will be augmented before training.
# augment_audio.py generates 5 variants per original clip:
#   pitch_up, pitch_down, stretch_fast, stretch_slow, noise
# Augmented files are saved in {word}_wrong/ alongside originals.
# build_metadata() picks them up automatically.
# Files are only created once — re-runs skip existing _aug_ files.
#
# Why augment:
#   Words with very few incorrect samples (<=8 original) give the SVC
#   almost nothing to learn the incorrect boundary from. Augmentation
#   creates acoustic variety while preserving the pronunciation error,
#   helping the SVC generalise to unseen wrong pronunciations.
#
# Results after augmentation (wrong_acc before -> after):
#   δραστηριότητες:  2 -> 0  (5  -> 29 incorrect samples)
#   λουλούδια:       0 -> 0  (2  -> 10) — stabilised
#   σκουπίδια:       0 -> 0  (2  -> 12) — stabilised
#   δεσμεύω:         2 -> 0  (7  -> 43)
#   ποίημα:          2 -> 0  (8  -> 48)
#   συμπληρώσουμε:   2 -> 0  (7  -> 43)
#   εκπαίδευση:      0 -> 0  (6  -> 36) — correct_rejected fixed too
#   ηλεκτρική:       0 -> 0  (4  -> 27) — stable
#   περιβάλλον:      0 -> 0  (2  -> 10) — correct% improved
#   υπεύθυνος:       0 -> 0  (6  -> 36) — correct_rejected fixed too
#
# NOT augmented:
#                imbalance, hurt correct% and correct_rejected
#   ταξιδεύουμε: already perfect (wrong_acc=0), augmentation dropped
#                correct% from 100% to 96.8% unnecessarily
#   βοοειδές:    53 incorrect samples + OC gate — enough already
#   all others:  sufficient incorrect samples (>20), wrong_acc=0
#
# To undo augmentation for a word: delete files containing '_aug_'
# in the {word}_wrong/ folder and remove the word from this set.
AUGMENT_WORDS = {
    # FULL & RAW
    'δραστηριότητες',
    'λουλούδια',
    'σκουπίδια',
    'δεσμεύω',
    'ποίημα',
    'συμπληρώσουμε',
    'εκπαίδευση',
    'ηλεκτρική',
    'περιβάλλον',
    'υπεύθυνος',
    'αλάτια', 
    'αυθαιρεσίες',
    'πρόσκληση',
    'σταφιδιασμένα',
    
    # RAW ONLY
    'περιβάλλον', 
    'πρόσκληση', 
    'σταφιδιασμένα', 
    'ταξιδεύουμε', 
    'παυσίπονο',
    'μετεγγραφές',
}


# ============================================================
# 11. DECISION RULE SETTINGS
# ============================================================

# Minimum correct_recall required during CV threshold selection.
# The best threshold must accept at least 75% of correct pronunciations.
# If no config meets this, the one with the best macro F1 is used instead.
MIN_CORRECT_RECALL = 0.75

# Phonetic score cutoffs tested during CV alongside probability thresholds.
# When a cutoff is active, samples with phonetic_score < cutoff are
# hard-rejected regardless of model probability.
# None = no cutoff (rely on model probability only).
PHONETIC_SCORE_CUTOFFS = [
    None,   # no cutoff — rely on model probability only
    80,     # reject anything below 80 phonetic score
    85,
    90,
    92,
    95,     # strict — only accept near-perfect phonetic matches
]

# Minimum model probability to auto-accept a correct prediction even when
# matched_transcription does not exactly match the target word.
# Used in Rule 3 of make_final_decision() — overrides transcription mismatch
# when the SVC is highly confident.
AUTO_ACCEPT_CONFIDENCE = 0.80


# ============================================================
# 12. ONE-CLASS SVM GATE PARAMETERS
# ============================================================
#
# The one-class SVM (OC-SVM) is trained on correct samples only.
# It provides a second opinion alongside the binary SVC by scoring
# how closely an audio resembles a correct pronunciation.
#
# The gate fires (sends auto-accept to possible_correct review) when
# BOTH conditions are true:
#   1. probability_correct < prob_threshold  (SVC not overwhelmingly certain)
#   2. oneclass_score < oc_threshold         (OC-SVM is skeptical)
#
# If probability_correct >= prob_threshold, the SVC is trusted regardless
# of the OC-SVM score.
#
# Thresholds are determined empirically from score distribution analysis
# of oneclass_scores_{word}.csv + predictions_{word}.csv.
# Only words where OC-SVM can cleanly separate wrong_accepted from correct
# samples are listed here.
#
# Words not listed have no gate applied.
#
# Current gate words and why:
#   ακαδημαϊκά:  30 incorrect samples, OC separates cleanly
#   επιχειρήματα: clean OC separation
#   περιβάλλον:  clean OC separation
#   ταξιδεύουμε: clean OC separation
#   μετεγγραφές: clean OC separation (updated threshold)
#   βοοειδές:    clean OC separation (new, after combined+PCA run)
#   εγκλείει:    limited but kept — partial separation
#
# REMOVED from gate (wrong_acc fixed by augmentation, gate no longer needed):
#   δραστηριότητες: augmentation fixed wrong_acc 2->0
#   δεσμεύω:        augmentation fixed wrong_acc 2->0
#   ποίημα:         augmentation fixed wrong_acc 2->0
#   συμπληρώσουμε:  augmentation fixed wrong_acc 2->0
#
# NOT included (OC signal not reliable enough):
#   λουλούδια:    39% correct cost for 1 wrong_accepted — too expensive
#   τήβεννος:     OC score not discriminating
#   παυσίπονο:    wrong samples have positive OC score — indistinguishable
ONECLASS_GATE_PARAMS = {
    "ακαδημαϊκά":  {"prob_threshold": 0.99, "oc_threshold": -0.172},
    "επιχειρήματα": {"prob_threshold": 0.99, "oc_threshold": -0.032},
    "περιβάλλον":  {"prob_threshold": 0.99, "oc_threshold": -0.023},
    "ταξιδεύουμε": {"prob_threshold": 0.93, "oc_threshold": -0.056},
    "μετεγγραφές": {"prob_threshold": 0.99, "oc_threshold": -0.020},
    "εγκλείει":    {"prob_threshold": 0.70, "oc_threshold": -0.009},
    "βοοειδές":    {"prob_threshold": 0.99, "oc_threshold": -0.240},
}