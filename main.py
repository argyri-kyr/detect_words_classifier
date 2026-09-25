# main.py

import os
import joblib
import pandas as pd

from globalVars import (
    WAV2VEC2_MODEL_NAME, WAV2VEC2_MODEL_PER_WORD,
    EMBEDDING_LAYER, EMBEDDING_LAYER_PER_WORD,
    EMBEDDING_LAYERS_COMBINE_PER_WORD,
    AUGMENT_WORDS,
)
import globalVars as gv


from augment_audio import augment_wrong_folder
from make_metadata import build_metadata
from extract_features import create_features_csv
from train import train_model
from evaluate import evaluate_model


def make_safe_folder_name(name):
    """
    Convert a name into a safe folder name.

    Examples:
        "lighteternal/wav2vec2-large-xlsr-53-greek"
        -> "lighteternal__wav2vec2-large-xlsr-53-greek"

        "word with spaces" -> "word_with_spaces"
    """

    safe = name.replace("/", "__")
    safe = safe.replace(" ", "_")
    return safe


def get_embedding_layer_for_word(word):
    """
    Returns the embedding layer to use for a specific word.

    If the word has a per-word override in globalVars.EMBEDDING_LAYER_PER_WORD,
    that layer is used. Otherwise the default EMBEDDING_LAYER is used.
    """

    if word in EMBEDDING_LAYER_PER_WORD:
        return EMBEDDING_LAYER_PER_WORD[word]
    return EMBEDDING_LAYER


def get_model_for_word(word):
    """
    Returns the wav2vec2 model to use for a specific word.

    Looks up WAV2VEC2_MODEL_PER_WORD first.
    Falls back to WAV2VEC2_MODEL_NAME if not found.
    """

    return WAV2VEC2_MODEL_PER_WORD.get(word, WAV2VEC2_MODEL_NAME)


def get_layer_folder_name(word):
    """
    Returns the folder name for the layer dimension of the output path.

    Combined-layer words get 'layer_combined'.
    Single-layer words get 'layer_N'.
    """

    if word in EMBEDDING_LAYERS_COMBINE_PER_WORD:
        return "layer_combined"
    return f"layer_{get_embedding_layer_for_word(word)}"


def collect_summary_row(word, model_name, safe_model_name, layer_folder):
    """
    Reads predictions_{word}.csv and model_oneclass.joblib for one word
    and returns a summary dict row. Returns None if predictions not found.
    """

    safe_word = make_safe_folder_name(word)

    predictions_path = os.path.join(
        "outputs", safe_model_name, layer_folder,
        safe_word, "report", f"predictions_{word}.csv"
    )

    if not os.path.exists(predictions_path):
        print(f"Warning: No predictions found for {word} ({layer_folder}), skipping.")
        return None

    df = pd.read_csv(predictions_path)

    correct_total   = len(df[df["true_label"] == "correct"])
    incorrect_total = len(df[df["true_label"] == "incorrect"])

    correct_accepted           = len(df[(df["true_label"] == "correct")   & (df["final_decision"] == "correct")])
    correct_possible_correct   = len(df[(df["true_label"] == "correct")   & (df["final_decision"] == "possible_correct")])
    correct_possible_incorrect = len(df[(df["true_label"] == "correct")   & (df["final_decision"] == "possible_incorrect")])
    correct_rejected           = len(df[(df["true_label"] == "correct")   & (df["final_decision"] == "incorrect")])

    wrong_rejected           = len(df[(df["true_label"] == "incorrect") & (df["final_decision"] == "incorrect")])
    wrong_possible_incorrect = len(df[(df["true_label"] == "incorrect") & (df["final_decision"] == "possible_incorrect")])
    wrong_possible_correct   = len(df[(df["true_label"] == "incorrect") & (df["final_decision"] == "possible_correct")])
    wrong_accepted           = len(df[(df["true_label"] == "incorrect") & (df["final_decision"] == "correct")])

    # OC threshold and interaction counts
    oneclass_threshold          = None
    oc_outside_svc_correct      = None
    oc_outside_review_incorrect = None
    oc_outside_review_correct   = None
    oc_inside_svc_incorrect     = None

    ocsvm_path = os.path.join(
        "outputs", safe_model_name, layer_folder,
        safe_word, "model_oneclass.joblib"
    )
    if os.path.exists(ocsvm_path):
        ocsvm_data = joblib.load(ocsvm_path)
        oneclass_threshold = ocsvm_data.get("oneclass_threshold", None)

    if oneclass_threshold is not None and "oneclass_score" in df.columns:
        oc_outside = df["oneclass_score"] < oneclass_threshold
        oc_inside  = df["oneclass_score"] >= oneclass_threshold
        oc_outside_svc_correct      = int((oc_outside & (df["final_decision"] == "correct")).sum())
        oc_outside_review_incorrect = int((oc_outside & (df["final_decision"] == "possible_incorrect")).sum())
        oc_outside_review_correct   = int((oc_outside & (df["final_decision"] == "possible_correct")).sum())
        oc_inside_svc_incorrect     = int((oc_inside  & (df["final_decision"] == "incorrect")).sum())

    # OC report
    oneclass_correct_pct   = None
    oneclass_incorrect_pct = None

    oneclass_report_path = os.path.join(
        "outputs", safe_model_name, layer_folder,
        safe_word, "report", f"oneclass_report_{word}.txt"
    )
    if os.path.exists(oneclass_report_path):
        with open(oneclass_report_path, "r", encoding="utf-8") as f:
            for line in f:
                if "True correct" in line and "(" in line:
                    try:
                        oneclass_correct_pct = float(line.split("(")[1].split("%")[0])
                    except (IndexError, ValueError):
                        pass
                if "True incorrect" in line and "(" in line:
                    try:
                        oneclass_incorrect_pct = float(line.split("(")[1].split("%")[0])
                    except (IndexError, ValueError):
                        pass

    return {
        "word":                         word,
        "model":                        model_name,
        "layer":                        layer_folder,
        "correct_total":                correct_total,
        "correct_auto_accepted":        correct_accepted,
        "correct_pct":                  round(correct_accepted / correct_total * 100, 1) if correct_total > 0 else 0,
        "correct_possible_correct":     correct_possible_correct,
        "correct_possible_incorrect":   correct_possible_incorrect,
        "correct_rejected":             correct_rejected,
        "incorrect_total":              incorrect_total,
        "incorrect_auto_rejected":      wrong_rejected,
        "incorrect_pct":                round(wrong_rejected / incorrect_total * 100, 1) if incorrect_total > 0 else 0,
        "incorrect_possible_incorrect": wrong_possible_incorrect,
        "incorrect_possible_correct":   wrong_possible_correct,
        "wrong_auto_accepted":          wrong_accepted,
        "oneclass_correct_pct":         oneclass_correct_pct,
        "oneclass_incorrect_pct":       oneclass_incorrect_pct,
        "oc_outside_svc_correct":       oc_outside_svc_correct,
        "oc_outside_review_incorrect":  oc_outside_review_incorrect,
        "oc_outside_review_correct":    oc_outside_review_correct,
        "oc_inside_svc_incorrect":      oc_inside_svc_incorrect,
    }


def run_pipeline_for_word(word, model_name):
    """
    Runs the full 4-step pipeline (metadata -> features -> train -> evaluate)
    for one word with its assigned model. Returns the summary row dict.
    """

    safe_word       = make_safe_folder_name(word)
    safe_model_name = make_safe_folder_name(model_name)
    embedding_layer = get_embedding_layer_for_word(word)
    layer_folder    = get_layer_folder_name(word)

    print(f"\n{'=' * 60}")
    print("  FULL PIPELINE")
    print(f"  word:  {word}")
    print(f"  model: {model_name}")
    print(f"  layer: {layer_folder}")
    print(f"{'=' * 60}\n")

    # --------------------------------------------------------
    # STEP 0 — Augment incorrect samples (if configured)
    # Runs before metadata so augmented clips are included
    # in build_metadata() on the same run.
    # Skips words not in AUGMENT_WORDS.
    # Skips files already augmented (_aug_ in filename).
    # --------------------------------------------------------
    if word in AUGMENT_WORDS:
        if word in ['περιβάλλον', 'πρόσκληση', 'σταφιδιασμένα', 'ταξιδεύουμε', 'παυσίπονο', 'μετεγγραφές', 'εκσυγχρονισμένος'] and gv.folder2 == "detect_words_decodewordspseudo_data_raw/":
            print(f"{'=' * 60}")
            print("  STEP 0 - Augment incorrect samples")
            print(f"{'=' * 60}\n")
            word_folder = os.path.join(gv.folder1, gv.folder2, word)
            augment_wrong_folder(word, word_folder)
        if word not in ['περιβάλλον', 'πρόσκληση', 'σταφιδιασμένα']:
            print(f"{'=' * 60}")
            print("  STEP 0 - Augment incorrect samples")
            print(f"{'=' * 60}\n")
            word_folder = os.path.join(gv.folder1, gv.folder2, word)
            augment_wrong_folder(word, word_folder)
            

    
    # STEP 1 - Create metadata CSV
    word_data_folder = os.path.join("data", safe_word)
    os.makedirs(word_data_folder, exist_ok=True)
    metadata_csv = os.path.join(word_data_folder, "metadata.csv")

    print(f"{'=' * 60}")
    print("  STEP 1 - Create metadata CSV")
    print(f"{'=' * 60}\n")

    build_metadata(words=[word], output_csv=metadata_csv)

    # Output paths
    model_output_folder = os.path.join(
        "outputs", safe_model_name, layer_folder, safe_word,
    )
    os.makedirs(model_output_folder, exist_ok=True)

    features_csv  = os.path.join(model_output_folder, "features.csv")
    model_path    = os.path.join(model_output_folder, "model.joblib")
    test_csv      = os.path.join(model_output_folder, "test.csv")
    report_folder = os.path.join(model_output_folder, "report")

    # STEP 2 - Extract features
    print(f"{'=' * 60}")
    print("  STEP 2 - Extract features")
    print(f"  layer: {layer_folder}")
    print(f"{'=' * 60}\n")

    original_layer = gv.EMBEDDING_LAYER
    if word not in EMBEDDING_LAYERS_COMBINE_PER_WORD:
        gv.EMBEDDING_LAYER = embedding_layer

    medoid_sequence = None
    try:
        medoid_sequence = create_features_csv(
            metadata_csv=metadata_csv,
            output_csv=features_csv,
            model_name=model_name,
        )
    finally:
        gv.EMBEDDING_LAYER = original_layer

    # STEP 3 - Train classifier
    print(f"\n{'=' * 60}")
    print("  STEP 3 - Train correct/incorrect classifier")
    print(f"{'=' * 60}\n")

    train_model(
        features_csv=features_csv,
        model_output=model_path,
        test_csv=test_csv,
        do_grid_search=True,
        medoid_sequence=medoid_sequence,
        embedding_layer=embedding_layer,
        model_name=model_name,      # ← add this
    )

    # STEP 4 - Evaluate classifier
    print(f"\n{'=' * 60}")
    print("  STEP 4 - Evaluate classifier")
    print(f"{'=' * 60}\n")

    evaluate_model(
        model_path=model_path,
        test_csv=test_csv,
        report_folder=report_folder,
        word=word,
    )

    print(f"\n{'=' * 60}")
    print(f"  DONE: {word} ({layer_folder})")
    print(f"{'=' * 60}\n")

    print(f"  Features CSV : {features_csv}")
    print(f"  Model        : {model_path}")
    print(f"  Test CSV     : {test_csv}")
    print(f"  Report       : {report_folder}")

    return collect_summary_row(word, model_name, safe_model_name, layer_folder)


def main():
    # ============================================================
    # SETTINGS
    # ============================================================

    words = [
        'ακαδημαϊκά', 'αλάτια', 'αρδευτικός', 'αυθαιρεσίες', 'βοοειδές', 'δεσμεύω',
        'δραστηριότητες', 'εγκλείει', 'εκπαίδευση', 'εκσυγχρονισμένος', 'επιχειρήματα',
        'ηλεκτρική', 'λουλούδια', 'μετεγγραφές', 'παυσίπονο', 'περιβάλλον', 'ποίημα',
        'πρόσκληση', 'σκουπίδια', 'σταφιδιασμένα', 'συμπληρώσουμε', 'ταξιδεύουμε',
        'τήβεννος', 'υπεύθυνος',
    ]

    # words = ['αρδευτικός']

    # ============================================================
    # Run pipeline for each word using its assigned model.
    # Model is read from WAV2VEC2_MODEL_PER_WORD in globalVars.py.
    # ============================================================
    summary_rows = []

    for word in words:

        model_name = get_model_for_word(word)
        row = run_pipeline_for_word(word, model_name)

        if row is not None:
            summary_rows.append(row)

    # ============================================================
    # Save combined summary
    # ============================================================
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join("outputs", "summary.csv")
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

        print(f"\n{'=' * 60}")
        print("  SUMMARY")
        print(f"{'=' * 60}\n")

        print(f"{'Word':<22} {'Model':<40} {'Layer':<16} {'Correct%':>9} {'Wrong_acc':>10} {'Correct_rej':>12}")
        print("-" * 105)
        for _, r in summary_df.iterrows():
            print(
                f"{r['word']:<22} {r['model']:<40} {r['layer']:<16} "
                f"{r['correct_pct']:>8.1f}% {r['wrong_auto_accepted']:>10} "
                f"{r['correct_rejected']:>12}"
            )

        print(f"\nSaved: {summary_path}")

    print(f"\n{'=' * 60}")
    print("  ALL WORDS FINISHED")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()