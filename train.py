# train.py
#
# Trains the binary SVC pronunciation classifier and the one-class SVM gate
# for each Greek pseudoword.
#
# Pipeline overview:
#   1. Load features CSV produced by extract_features.py
#   2. Split into train/test (80/20, stratified)
#   3. Grid search over SVC configs using only train split
#   4. Cross-validate to select best probability threshold + phonetic cutoff
#   5. Train final model on full train split
#   6. Save model.joblib (binary SVC) and model_oneclass.joblib (OC-SVM)
#   7. Save untouched test split for evaluate.py

from pathlib import Path
import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import globalVars as gv
from globalVars import (
    RANDOM_STATE,
    SVC_GRID,
    MIN_CORRECT_RECALL,
    PHONETIC_SCORE_CUTOFFS,
    TEST_SIZE_DEFAULT,
    TEST_SIZE_PER_WORD,
)

from utils import create_folder


# ============================================================
# FEATURE COLUMN SELECTION
# ============================================================

def get_feature_columns(df):
    """
    Extracts the names of all feature columns from the DataFrame.

    Feature columns produced by extract_features.py are always
    named f_0, f_1, f_2, ... so we select all columns that start
    with "f_". Metadata columns (word, label, phonetic_score, etc.)
    are excluded automatically.

    Returns:
        List of column name strings, e.g. ["f_0", "f_1", ..., "f_6144"]
    """

    feature_columns = []

    for column in df.columns:
        if column.startswith("f_"):
            feature_columns.append(column)

    return feature_columns


# ============================================================
# CLASSIFIER PIPELINE
# ============================================================

def make_classifier(kernel="rbf", C=5, gamma="scale"):
    """
    Builds the full sklearn Pipeline used as the binary SVC classifier.

    The pipeline has three sequential steps:

    Step 1 — PCA (n_components=0.95):
        Reduces the feature space while keeping 95% of variance.
        For combined-layer words, the input is 6144 features
        (3 layers x 2048 mean+std). PCA compresses this to typically
        50-150 principal components. This prevents overfitting when
        dataset size is small (80-150 samples per word).
        PCA runs BEFORE the scaler because it needs unscaled data
        to compute variance directions correctly.

    Step 2 — StandardScaler:
        Normalizes each PCA component to zero mean and unit variance.
        SVC is sensitive to feature scale — normalization ensures no
        single component dominates the kernel distances.

    Step 3 — SVC (Support Vector Classifier):
        Classifies each sample as correct (1) or incorrect (0).
        probability=True enables predict_proba() which returns
        a confidence score (0-1) used by the decision rules.
        class_weight='balanced' adjusts for unequal class counts
        by upweighting the minority class — no data removal.

    Returns:
        sklearn Pipeline that can be fit/predict like any estimator.
    """

    classifier = Pipeline([
        ("pca", PCA(n_components=0.95, random_state=RANDOM_STATE)),
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ])

    return classifier


# ============================================================
# DECISION RULE (used inside CV only)
# ============================================================

def apply_decision_rule(
    probability_correct,
    phonetic_scores,
    probability_threshold,
    phonetic_cutoff
):
    """
    Converts model probabilities into binary predictions (0/1).

    Used only inside cross-validation to evaluate different
    threshold/cutoff combinations. The real decision logic used
    at evaluation and prediction time is in decision_rules.py
    (which supports 4 outputs: correct/incorrect/possible_correct/
    possible_incorrect). This simplified version produces only
    binary 0/1 for CV metric computation.

    Logic per sample:
        1. If phonetic_cutoff is set AND phonetic_score < cutoff:
               → predict 0 (incorrect), ignoring model probability.
               This hard-rejects samples with poor phonetic alignment
               regardless of what the SVC thinks.
        2. Otherwise:
               → predict 1 (correct) if probability >= threshold
               → predict 0 (incorrect) otherwise

    Args:
        probability_correct: list of floats (SVC output, 0-1)
        phonetic_scores:     list of floats (from Excel aligner)
        probability_threshold: float, above which we predict correct
        phonetic_cutoff:     float or None, below which we force incorrect

    Returns:
        List of int predictions (0=incorrect, 1=correct)
    """

    predictions = []

    for i, prob in enumerate(probability_correct):
        prediction = None

        # Step 1: phonetic cutoff hard-reject
        if phonetic_cutoff is not None:
            score = phonetic_scores[i]
            if pd.notna(score) and score < phonetic_cutoff:
                prediction = 0   # force incorrect regardless of SVC

        # Step 2: probability threshold
        if prediction is None:
            if prob >= probability_threshold:
                prediction = 1   # correct
            else:
                prediction = 0   # incorrect

        predictions.append(prediction)

    return predictions


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):
    """
    Computes all classification metrics for one set of predictions.

    Computed for both classes (correct=1, incorrect=0):
        - accuracy:           overall fraction correct
        - macro_f1:           average F1 across both classes
        - incorrect_precision: of all predicted-incorrect, how many truly are
        - incorrect_recall:   of all true-incorrect, how many we caught
        - incorrect_f1:       harmonic mean of above two for incorrect class
        - correct_precision:  of all predicted-correct, how many truly are
        - correct_recall:     of all true-correct, how many we accepted
        - correct_f1:         harmonic mean of above two for correct class

    Why macro_f1 matters:
        With imbalanced classes (more correct than incorrect samples),
        accuracy alone is misleading. Macro F1 gives equal weight to
        both classes regardless of their size.

    Returns:
        dict of metric_name -> float value
    """

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),

        "macro_f1": f1_score(
            y_true, y_pred,
            average="macro",
            zero_division=0
        ),

        "incorrect_precision": precision_score(
            y_true, y_pred,
            pos_label=0,
            zero_division=0
        ),

        "incorrect_recall": recall_score(
            y_true, y_pred,
            pos_label=0,
            zero_division=0
        ),

        "incorrect_f1": f1_score(
            y_true, y_pred,
            pos_label=0,
            zero_division=0
        ),

        "correct_precision": precision_score(
            y_true, y_pred,
            pos_label=1,
            zero_division=0
        ),

        "correct_recall": recall_score(
            y_true, y_pred,
            pos_label=1,
            zero_division=0
        ),

        "correct_f1": f1_score(
            y_true, y_pred,
            pos_label=1,
            zero_division=0
        ),
    }

    return metrics


# ============================================================
# CUTOFF KEY HELPERS
# ============================================================

def cutoff_to_key(cutoff):
    """
    Converts a phonetic cutoff value to a string key for CSV grouping.

    None  -> "None"  (no cutoff applied)
    95.0  -> "95.0"

    Needed because pandas groupby cannot handle None in mixed columns.
    """

    if cutoff is None:
        return "None"
    return str(cutoff)


def key_to_cutoff(cutoff_key):
    """
    Converts a string cutoff key back to its numeric value or None.

    "None" -> None
    "95.0" -> 95.0

    Used after reading the best cutoff back from the CV summary DataFrame.
    """

    if cutoff_key == "None":
        return None
    return float(cutoff_key)


# ============================================================
# THRESHOLD SEARCH
# ============================================================

def find_best_threshold(y_true, probability_correct):
    """
    Searches for the probability threshold that maximises macro F1.

    Tries 19 threshold values from 0.05 to 0.95 in steps of 0.05.
    For each threshold, converts probabilities to binary predictions
    and computes macro F1.

    This function is a standalone utility — the main threshold selection
    happens inside run_cross_validation() across folds.

    Returns:
        float: the threshold value with the highest macro F1
    """

    thresholds = [0.05 + i * 0.05 for i in range(19)]

    best_threshold = 0.50
    best_macro_f1  = -1

    for threshold in thresholds:
        y_pred = [1 if prob >= threshold else 0 for prob in probability_correct]

        macro_f1 = f1_score(
            y_true, y_pred,
            average="macro",
            zero_division=0
        )

        if macro_f1 > best_macro_f1:
            best_macro_f1  = macro_f1
            best_threshold = threshold

    return best_threshold


# ============================================================
# CROSS-VALIDATION
# ============================================================

def run_cross_validation(df, feature_columns, cv_output_csv, svc_params):
    """
    Runs 5-fold stratified cross-validation to select the best
    probability threshold and phonetic score cutoff.

    Why CV for threshold selection:
        The threshold and cutoff are hyperparameters. Selecting them
        on the test set would leak information. CV uses only the
        training split so the test set remains unseen.

    What it does:
        - Splits the training data into 5 folds (stratified by label).
        - For each fold: trains the classifier on 4 folds, evaluates
          on the remaining fold.
        - Tests all combinations of probability thresholds (0.05..0.95)
          and phonetic cutoffs (from PHONETIC_SCORE_CUTOFFS).
        - Averages metrics across all 5 folds for each combination.
        - Selects the combination that:
              1. Achieves correct_recall >= MIN_CORRECT_RECALL (0.75)
              2. Among those, maximises incorrect_recall
              3. Then correct_recall
              4. Then macro_f1

    Selection priority explained:
        We require at least 75% of correct pronunciations to be
        accepted (correct_recall >= 0.75). Among all configs that
        meet this, we prefer the one that catches the most incorrect
        pronunciations (highest incorrect_recall), because wrong_accepted
        = 0 is the primary goal.

    Returns:
        dict with keys:
            fold_results:       raw per-fold metrics DataFrame
            summary_results:    averaged metrics DataFrame
            best_decision_params: dict with probability_threshold,
                                  phonetic_cutoff, selection_reason
            best_metrics:       dict of averaged metric values
    """

    X = df[feature_columns]
    y = df["label_id"]

    # Determine available phonetic cutoffs
    has_phonetic_score = "phonetic_score" in df.columns
    phonetic_cutoffs   = PHONETIC_SCORE_CUTOFFS if has_phonetic_score else [None]

    # Fine-grained threshold grid: 0.05, 0.10, ..., 0.95
    probability_thresholds = [0.05 + i * 0.05 for i in range(19)]

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    rows = []

    for fold_number, (train_index, val_index) in enumerate(cv.split(X, y), start=1):

        # Split this fold into train and validation
        train_df = df.iloc[train_index]
        val_df   = df.iloc[val_index]

        X_train = train_df[feature_columns]
        y_train = train_df["label_id"]
        y_val   = val_df["label_id"]

        # Train classifier on this fold's training portion
        classifier = make_classifier(
            kernel=svc_params["kernel"],
            C=svc_params["C"],
            gamma=svc_params["gamma"]
        )
        classifier.fit(X_train, y_train)

        # Get probability predictions on validation fold
        probability_correct = classifier.predict_proba(val_df[feature_columns])[:, 1]

        phonetic_scores = (
            val_df["phonetic_score"].values
            if has_phonetic_score
            else np.full(len(val_df), np.nan)
        )

        # Test all threshold x cutoff combinations on this fold
        for probability_threshold in probability_thresholds:
            for phonetic_cutoff in phonetic_cutoffs:

                y_pred = apply_decision_rule(
                    probability_correct=probability_correct,
                    phonetic_scores=phonetic_scores,
                    probability_threshold=probability_threshold,
                    phonetic_cutoff=phonetic_cutoff
                )

                metrics = calculate_metrics(y_true=y_val, y_pred=y_pred)

                rows.append({
                    "fold":                fold_number,
                    "probability_threshold": probability_threshold,
                    "phonetic_cutoff":     cutoff_to_key(phonetic_cutoff),
                    "kernel":              svc_params["kernel"],
                    "C":                   svc_params["C"],
                    "gamma":               svc_params["gamma"],
                    **metrics,
                })

    results_df = pd.DataFrame(rows)

    # Save raw fold-level results
    cv_output_csv = Path(cv_output_csv)
    create_folder(cv_output_csv.parent)
    results_df.to_csv(cv_output_csv, index=False)

    # Average metrics across all 5 folds for each threshold/cutoff combo
    summary_df = (
        results_df
        .groupby(
            ["probability_threshold", "phonetic_cutoff", "kernel", "C", "gamma"],
            dropna=False
        )
        [["accuracy", "macro_f1",
          "incorrect_precision", "incorrect_recall", "incorrect_f1",
          "correct_precision",   "correct_recall",   "correct_f1"]]
        .mean()
        .reset_index()
    )

    # Select best threshold/cutoff combination
    valid = summary_df[summary_df["correct_recall"] >= MIN_CORRECT_RECALL]

    if len(valid) > 0:
        # Among configs meeting the recall floor, maximise incorrect detection
        best_row = valid.sort_values(
            by=["incorrect_recall", "correct_recall", "macro_f1"],
            ascending=False
        ).iloc[0]
        selection_reason = (
            f"correct_recall >= {MIN_CORRECT_RECALL}, "
            "then best incorrect_recall, then best correct_recall, then best macro_f1"
        )
    else:
        # Fallback: no config met the floor — pick best macro F1 overall
        best_row = summary_df.sort_values(by=["macro_f1"], ascending=False).iloc[0]
        selection_reason = (
            "no option reached minimum correct recall, selected best macro_f1"
        )

    best_decision_params = {
        "probability_threshold": float(best_row["probability_threshold"]),
        "phonetic_cutoff":       key_to_cutoff(best_row["phonetic_cutoff"]),
        "selection_reason":      selection_reason,
    }

    best_metrics = {
        "accuracy":            float(best_row["accuracy"]),
        "macro_f1":            float(best_row["macro_f1"]),
        "incorrect_precision": float(best_row["incorrect_precision"]),
        "incorrect_recall":    float(best_row["incorrect_recall"]),
        "incorrect_f1":        float(best_row["incorrect_f1"]),
        "correct_precision":   float(best_row["correct_precision"]),
        "correct_recall":      float(best_row["correct_recall"]),
        "correct_f1":          float(best_row["correct_f1"]),
    }

    print("=" * 60)
    print("  CROSS-VALIDATION BEST DECISION")
    print("=" * 60)
    print()
    print(f"SVC parameters:          {svc_params}")
    print(f"Probability threshold:   {best_decision_params['probability_threshold']}")
    print(f"Phonetic cutoff:         {best_decision_params['phonetic_cutoff']}")
    print(f"Selection reason:        {selection_reason}")
    print()
    print(f"accuracy:                {best_metrics['accuracy']:.3f}")
    print(f"macro_f1:                {best_metrics['macro_f1']:.3f}")
    print(f"incorrect_precision:     {best_metrics['incorrect_precision']:.3f}")
    print(f"incorrect_recall:        {best_metrics['incorrect_recall']:.3f}")
    print(f"incorrect_f1:            {best_metrics['incorrect_f1']:.3f}")
    print(f"correct_precision:       {best_metrics['correct_precision']:.3f}")
    print(f"correct_recall:          {best_metrics['correct_recall']:.3f}")
    print(f"correct_f1:              {best_metrics['correct_f1']:.3f}")
    print()
    print(f"Saved CV details:        {cv_output_csv}")
    print()

    return {
        "fold_results":       results_df,
        "summary_results":    summary_df,
        "best_decision_params": best_decision_params,
        "best_metrics":       best_metrics,
    }


# ============================================================
# SVC GRID SEARCH
# ============================================================

def grid_search_svc(df, feature_columns, output_folder):
    """
    Searches over a small grid of SVC hyperparameter configurations.

    For each configuration (kernel, C, gamma), runs full
    cross-validation to find the best threshold and cutoff.
    Then selects the configuration with the best CV performance.

    Grid is defined in globalVars.SVC_GRID. Current configs test
    RBF kernel with different C and gamma values, plus linear kernel.

    Selection priority (same as run_cross_validation):
        1. correct_recall >= MIN_CORRECT_RECALL
        2. Best incorrect_recall among valid configs
        3. Best correct_recall
        4. Best macro_f1

    Why grid search:
        Different words have different decision boundaries. The optimal
        C (regularisation) and gamma (RBF width) vary per word.
        Grid search finds the best config without manual tuning.

    Returns:
        best_svc_params:     dict with kernel, C, gamma
        best_decision_params: dict with probability_threshold, phonetic_cutoff
    """

    print("=" * 60)
    print("  SMALL SVC GRID SEARCH")
    print("=" * 60)
    print()
    print(f"Minimum correct recall: {MIN_CORRECT_RECALL}")
    print(f"Number of configs:      {len(SVC_GRID)}")
    print()

    all_results = []

    for i, params in enumerate(SVC_GRID, start=1):
        print(f"Testing config {i}/{len(SVC_GRID)}: {params}")

        temp_csv = Path(output_folder) / f"grid_config_{i}.csv"

        # Run full CV for this config to get best threshold/cutoff
        cv_result = run_cross_validation(
            df=df,
            feature_columns=feature_columns,
            cv_output_csv=temp_csv,
            svc_params=params
        )

        best_metrics  = cv_result["best_metrics"]
        best_decision = cv_result["best_decision_params"]

        all_results.append({
            "config_id":           i,
            "kernel":              params["kernel"],
            "C":                   params["C"],
            "gamma":               params["gamma"],
            "probability_threshold": best_decision["probability_threshold"],
            "phonetic_cutoff":     best_decision["phonetic_cutoff"],
            **best_metrics,
        })

    results_df = pd.DataFrame(all_results)

    # Save grid search results for inspection
    grid_results_csv = Path(output_folder) / "svc_grid_results.csv"
    results_df.to_csv(grid_results_csv, index=False)

    # Select best config using the same priority as CV selection
    valid = results_df[results_df["correct_recall"] >= MIN_CORRECT_RECALL]

    if len(valid) > 0:
        best_row = valid.sort_values(
            by=["incorrect_recall", "correct_recall", "macro_f1"],
            ascending=False
        ).iloc[0]
        selection_reason = (
            f"correct_recall >= {MIN_CORRECT_RECALL}, "
            "then best incorrect_recall, then best correct_recall, then best macro_f1"
        )
    else:
        best_row = results_df.sort_values(by=["macro_f1"], ascending=False).iloc[0]
        selection_reason = (
            "no config reached minimum correct recall, selected best macro_f1"
        )

    best_svc_params = {
        "kernel": best_row["kernel"],
        "C":      int(best_row["C"]) if str(best_row["C"]).isdigit() else best_row["C"],
        "gamma":  best_row["gamma"],
    }

    best_decision_params = {
        "probability_threshold": float(best_row["probability_threshold"]),
        "phonetic_cutoff": (
            None
            if pd.isna(best_row["phonetic_cutoff"])
            else float(best_row["phonetic_cutoff"])
        ),
        "selection_reason": selection_reason,
    }

    print()
    print("=" * 60)
    print("  GRID SEARCH SUMMARY")
    print("=" * 60)
    print()

    display_columns = [
        "config_id", "kernel", "C", "gamma",
        "probability_threshold", "phonetic_cutoff",
        "accuracy", "macro_f1",
        "incorrect_precision", "incorrect_recall", "incorrect_f1",
        "correct_recall", "correct_f1",
    ]
    print(results_df[display_columns].sort_values(
        by=["correct_recall", "incorrect_f1"], ascending=False
    ))

    print()
    print("Selected best SVC params:")
    print(best_svc_params)
    print()
    print("Selected best decision params:")
    print(best_decision_params)
    print()
    print(f"Saved grid results:   {grid_results_csv}")
    print()

    return best_svc_params, best_decision_params


# ============================================================
# MAIN TRAINING ENTRY POINT
# ============================================================

def train_model(
    features_csv,
    model_output,
    test_csv,
    do_grid_search=False,
    medoid_sequence=None,
    embedding_layer=None,
    model_name=None,
):
    """
    Full training pipeline for one word. Entry point called by main.py.

    Steps (strictly ordered to avoid data leakage):

        Step 1 — Load features CSV
            Reads features produced by extract_features.py.
            Each row is one audio clip; columns are f_0..f_N plus metadata.

        Step 2 — Train/test split (80/20, stratified)
            Split happens FIRST before any parameter selection.
            The test set is saved untouched and never seen during training.
            Stratification ensures both splits have the same class ratio.

        Step 3 — Grid search + CV (on train split only)
            If do_grid_search=True: tests all SVC configs from SVC_GRID.
            If False: uses default RBF config.
            CV runs only on train split — test split never touched.
            Selects best kernel/C/gamma and best threshold/cutoff.

        Step 4 — Final CV (on train split only)
            One more CV pass with the selected SVC config to confirm
            the best threshold/cutoff and get honest validation estimates.

        Step 5 — Train final model on full train split
            The Pipeline (PCA + scaler + SVC) is fitted on all train samples.
            No validation held out at this stage.

        Step 6 — Save model.joblib
            Saves: trained Pipeline, feature_columns, best_threshold,
            best_phonetic_cutoff, embedding_layer, medoid_sequence,
            model_name (wav2vec2 model used for features).
            model_name is saved so predict.py can load the correct
            wav2vec2 model at inference time without reading globalVars.

        Step 7 — Save test_csv
            Untouched 20% test split saved for evaluate.py.

        Step 8 — Train one-class SVM
            See train_one_class_model() below.

    Args:
        features_csv:    path to features.csv from extract_features.py
        model_output:    path where model.joblib will be saved
        test_csv:        path where test split CSV will be saved
        do_grid_search:  if True, run SVC grid search; else use default
        medoid_sequence: mel-spectrogram medoid (if USE_MEL_PROTOTYPE)
        embedding_layer: wav2vec2 layer used for single-layer words
        model_name:      wav2vec2 model name used to extract features
    """

    print("\n============================================================")
    print("  TRAINING")
    print("============================================================\n")

    # Step 1: Load features
    df = pd.read_csv(features_csv)
    print("Dataset label counts:")
    print(df["label"].value_counts())

    feature_columns = get_feature_columns(df)
    if len(feature_columns) == 0:
        raise ValueError("No feature columns found. Expected columns starting with 'f_'.")
    if df["label_id"].nunique() < 2:
        raise ValueError("Need both classes: correct and incorrect.")

    # Step 2: Train/test split — test set is never seen during training
    # Per-word test size — small datasets benefit from more training samples
    test_size = TEST_SIZE_PER_WORD.get(
        df["word"].iloc[0] if "word" in df.columns else "",
        TEST_SIZE_DEFAULT
    )

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=df["label_id"],
    )

    print("\nTrain label counts:")
    print(train_df["label"].value_counts())
    print("\nTest label counts:")
    print(test_df["label"].value_counts())

    output_folder = str(Path(model_output).parent)
    create_folder(output_folder)

    # Step 3: Select SVC params and threshold/cutoff via CV on train split
    if do_grid_search:
        best_svc_params, best_decision_params = grid_search_svc(
            df=train_df,
            feature_columns=feature_columns,
            output_folder=output_folder,
        )
    else:
        # Use default RBF config, still run CV to select threshold/cutoff
        best_svc_params = {"kernel": "rbf", "C": 5, "gamma": "scale"}
        cv_output_csv   = str(Path(output_folder) / "cv_results.csv")
        final_cv_result = run_cross_validation(
            df=train_df,
            feature_columns=feature_columns,
            cv_output_csv=cv_output_csv,
            svc_params=best_svc_params,
        )
        best_decision_params = final_cv_result["best_decision_params"]

    # Step 4: Final CV pass to confirm threshold/cutoff with selected config
    cv_output_csv   = str(Path(output_folder) / "cv_results.csv")
    final_cv_result = run_cross_validation(
        df=train_df,
        feature_columns=feature_columns,
        cv_output_csv=cv_output_csv,
        svc_params=best_svc_params,
    )
    best_decision_params = final_cv_result["best_decision_params"]

    # Step 5: Train final Pipeline on full train split
    X_train = train_df[feature_columns]
    y_train = train_df["label_id"]
    model   = make_classifier(**best_svc_params)
    model.fit(X_train, y_train)

    # Step 6: Save model.joblib
    # model_name is saved so predict.py knows which wav2vec2 model
    # was used to extract features — critical for correct inference.
    model_data = {
        "model":                    model,
        "feature_columns":          feature_columns,
        "model_type":               "binary",
        "svc_params":               best_svc_params,
        "best_threshold":           best_decision_params["probability_threshold"],
        "best_phonetic_cutoff":     best_decision_params["phonetic_cutoff"],
        "decision_selection_reason": best_decision_params["selection_reason"],
        "embedding_layer":          embedding_layer if embedding_layer is not None else gv.EMBEDDING_LAYER,
        "medoid_sequence":          medoid_sequence,
        "model_name":               model_name if model_name is not None else gv.WAV2VEC2_MODEL_NAME,
    }
    joblib.dump(model_data, model_output)

    # Step 7: Save untouched test split for evaluate.py
    test_df.to_csv(test_csv, index=False, encoding="utf-8-sig")

    # Step 8: Train one-class SVM on correct training samples
    train_one_class_model(
        train_df=train_df,
        feature_columns=feature_columns,
        output_folder=output_folder,
    )

    print("\n============================================================")
    print("  FINAL MODEL SAVED")
    print("============================================================\n")
    print(f"Final SVC parameters:       {best_svc_params}")
    print(f"Probability threshold:      {best_decision_params['probability_threshold']}")
    print(f"Phonetic cutoff:            {best_decision_params['phonetic_cutoff']}")
    print(f"Decision selection reason:  {best_decision_params['selection_reason']}")
    print(f"wav2vec2 model:             {model_data['model_name']}")
    print(f"\nModel saved:                {model_output}")
    print(f"Test split saved:           {test_csv}")


# ============================================================
# ONE-CLASS SVM TRAINING
# ============================================================

def train_one_class_model(train_df, feature_columns, output_folder):
    """
    Trains a One-Class SVM (OC-SVM) on correct pronunciation samples only.

    Purpose:
        The binary SVC sometimes auto-accepts wrong pronunciations when
        they happen to fall near the correct cluster. The OC-SVM provides
        a second opinion: it learns what correct pronunciations sound like
        and flags anything that doesn't match as suspicious.

    How it works:
        - Trained ONLY on correct samples from the training split.
        - At evaluation/prediction time, it outputs a signed score
          (decision_function): positive = inside correct boundary,
          negative = outside (sounds wrong).
        - The gate threshold is the 5th percentile of correct training
          scores: samples below this are flagged even if they are correct,
          but only 5% of them should be at most.

    Nu parameter search:
        Nu controls how tightly the boundary wraps around correct samples.
        Lower nu = tighter boundary (fewer training errors allowed).
        We try [0.01, 0.05, 0.10, 0.20] and pick the one that:
            1. Keeps correct_recall >= 0.75 (doesn't reject too many correct)
            2. Maximises incorrect_detection (catches the most wrong ones)

    Saved to model_oneclass.joblib:
        - model:              fitted OC-SVM Pipeline (scaler + OneClassSVM)
        - feature_columns:    same columns used in binary SVC training
        - best_nu:            selected nu value
        - oneclass_threshold: 5th percentile of correct training scores,
                              used as gate boundary in decision_rules.py
                              and ONECLASS_GATE_PARAMS in globalVars.py

    Note: OC-SVM does NOT use PCA. It operates directly on raw features
    with its own StandardScaler. This keeps it independent from the
    binary SVC pipeline.
    """

    from sklearn.svm import OneClassSVM
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    # Use only correct samples for training the boundary
    correct_train = train_df[train_df["label"] == "correct"]
    X_correct     = correct_train[feature_columns]

    nu_values = [0.01, 0.05, 0.10, 0.20]
    results   = []

    for nu in nu_values:
        # Build and fit OC-SVM with this nu
        ocsvm = Pipeline([
            ("scaler", StandardScaler()),
            ("ocsvm", OneClassSVM(kernel="rbf", nu=nu, gamma="scale"))
        ])
        ocsvm.fit(X_correct)

        # Evaluate on full training set to see how well it separates classes
        preds = ocsvm.predict(train_df[feature_columns])
        # +1 = inlier (predicted correct), -1 = outlier (predicted wrong)

        correct_mask   = train_df["label"] == "correct"
        incorrect_mask = train_df["label"] == "incorrect"

        correct_detected   = (preds[correct_mask]   == 1).sum()
        correct_total      = correct_mask.sum()
        incorrect_detected = (preds[incorrect_mask] == -1).sum()
        incorrect_total    = incorrect_mask.sum()

        results.append({
            "nu":                  nu,
            "correct_recall":      correct_detected / correct_total      if correct_total > 0      else 0,
            "incorrect_detection": incorrect_detected / incorrect_total  if incorrect_total > 0    else 0,
        })

    results_df = pd.DataFrame(results)

    # Select best nu: maximise incorrect_detection while keeping correct_recall >= 0.75
    valid = results_df[results_df["correct_recall"] >= 0.75]
    if len(valid) > 0:
        best = valid.sort_values("incorrect_detection", ascending=False).iloc[0]
    else:
        best = results_df.sort_values("incorrect_detection", ascending=False).iloc[0]

    best_nu = best["nu"]

    # Retrain on all correct training samples with the selected nu
    final_ocsvm = Pipeline([
        ("scaler", StandardScaler()),
        ("ocsvm", OneClassSVM(kernel="rbf", nu=best_nu, gamma="scale"))
    ])
    final_ocsvm.fit(X_correct)

    # Compute gate threshold: 5th percentile of correct training scores.
    # Samples below this threshold are flagged by the gate.
    # At most 5% of true correct samples should fall below it.
    correct_scores     = final_ocsvm.decision_function(X_correct)
    oneclass_threshold = float(np.percentile(correct_scores, 5))

    # Save OC-SVM with threshold
    ocsvm_path = os.path.join(output_folder, "model_oneclass.joblib")
    joblib.dump({
        "model":              final_ocsvm,
        "feature_columns":    feature_columns,
        "best_nu":            best_nu,
        "oneclass_threshold": oneclass_threshold,
    }, ocsvm_path)

    print(f"\n{'=' * 60}")
    print("  ONE-CLASS SVM RESULTS (trained on correct only)")
    print(f"{'=' * 60}\n")
    print(results_df.to_string(index=False))
    print(f"\nSelected nu:          {best_nu}")
    print(f"Oneclass threshold:   {oneclass_threshold:.5f}")
    print(f"Saved: {ocsvm_path}")

    return final_ocsvm, feature_columns, best_nu