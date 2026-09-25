
# evaluate.py


import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from globalVars import ID_TO_LABEL
from utils import create_folder
from decision_rules import make_final_decision


def evaluate_model(model_path, test_csv, report_folder, word):
    """
    Evaluates the trained classifier.

    This function reports three views:

    1. Semantic metrics:
        correct              -> correct
        possible_correct     -> correct
        incorrect            -> incorrect
        possible_incorrect   -> incorrect

    2. Automatic-acceptance safety metrics:
        correct              -> accepted automatically
        everything else      -> not automatically accepted

    3. Decided-only metrics:
        possible_correct and possible_incorrect are EXCLUDED entirely.
        Coverage shows what fraction of samples got an automatic decision.

    4. Per-class outcome breakdown:
        For each true label, shows what fraction ended up in each
        final_decision category. possible_* is its own "review" slice,
        not folded into correct or incorrect.
    """

    create_folder(report_folder)

    # ------------------------------------------------------------
    # Load trained classifier and saved decision parameters
    # ------------------------------------------------------------
    model_data = joblib.load(model_path)

    model = model_data["model"]
    feature_columns = model_data["feature_columns"]

    best_threshold = model_data["best_threshold"]
    best_phonetic_cutoff = model_data["best_phonetic_cutoff"]

    # ------------------------------------------------------------
    # Load one-class model for gate scoring (if it exists)
    # ------------------------------------------------------------
    ocsvm_model = None
    ocsvm_feature_columns = None

    ocsvm_path_gate = os.path.normpath(
        os.path.join(report_folder, "..", "model_oneclass.joblib")
    )

    if os.path.exists(ocsvm_path_gate):
        ocsvm_gate_data = joblib.load(ocsvm_path_gate)
        ocsvm_model = ocsvm_gate_data["model"]
        ocsvm_feature_columns = ocsvm_gate_data["feature_columns"]
        
    # ------------------------------------------------------------
    # Load test data
    # ------------------------------------------------------------
    test_df = pd.read_csv(test_csv)

    X_test = test_df[feature_columns]
    y_true = test_df["label_id"].tolist()

    # ------------------------------------------------------------
    # Probability that each audio is correct according to SVC
    # ------------------------------------------------------------
    probability_correct = model.predict_proba(X_test)[:, 1]

    results = test_df.copy()
    results["probability_correct"] = probability_correct

    model_predictions = []
    final_decisions = []
    decision_reasons = []
    transcription_exact_matches = []
    review_required_values = []
    oneclass_scores = []         

    # ------------------------------------------------------------
    # Apply the shared safe decision rule to every test sample
    # ------------------------------------------------------------
    for _, row in results.iterrows():

        # Compute one-class score for gate
        oneclass_score = None
        if ocsvm_model is not None and ocsvm_feature_columns is not None:
            try:
                X_row = pd.DataFrame([row])[ocsvm_feature_columns]
                oneclass_score = float(
                    ocsvm_model.decision_function(X_row)[0]
                )
            except Exception:
                oneclass_score = None
                
        decision = make_final_decision(
            target_word=row["word"],
            probability_correct=row["probability_correct"],
            phonetic_score=row["phonetic_score"],
            probability_threshold=best_threshold,
            phonetic_cutoff=best_phonetic_cutoff,
            matched_transcription=row.get("matched_transcription", None),
            detected_status=row.get("detected_status", None),
            whisper_similarity=row.get("whisper_similarity", None),
            whisper_transcription=row.get("whisper_transcription", None),
            oneclass_score=oneclass_score,
        )
        model_predictions.append(decision["model_prediction"])
        final_decisions.append(decision["final_decision"])
        decision_reasons.append(decision["decision_reason"])
        transcription_exact_matches.append(decision["transcription_exact_match"])
        review_required_values.append(decision["review_required"])
        oneclass_scores.append(oneclass_score)

    results["model_prediction"] = model_predictions
    results["final_decision"] = final_decisions
    results["decision_reason"] = decision_reasons
    results["transcription_exact_match"] = transcription_exact_matches
    results["review_required"] = review_required_values
    results["oneclass_score"] = oneclass_scores

    results["true_label"] = results["label_id"].map(ID_TO_LABEL)

    # ------------------------------------------------------------
    # Semantic binary prediction.
    #
    # Meaning:
    #   correct              -> correct
    #   possible_correct     -> correct
    #   incorrect            -> incorrect
    #   possible_incorrect   -> incorrect
    # ------------------------------------------------------------
    results["semantic_binary_prediction"] = results["final_decision"].map(
        {
            "correct": "correct",
            "possible_correct": "correct",
            "incorrect": "incorrect",
            "possible_incorrect": "incorrect",
        }
    )

    y_pred_semantic = results["semantic_binary_prediction"].map(
        {"incorrect": 0, "correct": 1}
    ).tolist()

    semantic_report = classification_report(
        y_true,
        y_pred_semantic,
        target_names=["incorrect", "correct"],
        digits=2,
    )

    # ------------------------------------------------------------
    # Automatic-acceptance binary prediction.
    #
    # Meaning:
    #   correct              -> correct
    #   incorrect            -> incorrect
    #   possible_correct     -> incorrect
    #   possible_incorrect   -> incorrect
    #
    # Only final_decision == "correct" is automatically accepted.
    # ------------------------------------------------------------
    results["automatic_acceptance_prediction"] = results["final_decision"].apply(
        lambda d: "correct" if d == "correct" else "incorrect"
    )

    y_pred_auto = results["automatic_acceptance_prediction"].map(
        {"incorrect": 0, "correct": 1}
    ).tolist()

    automatic_report = classification_report(
        y_true,
        y_pred_auto,
        target_names=["incorrect", "correct"],
        digits=2,
    )

    # ------------------------------------------------------------
    # Decided-only metrics.
    #
    # possible_correct and possible_incorrect are EXCLUDED entirely.
    # They are not counted as correct OR incorrect, because they
    # require manual listening to decide.
    # ------------------------------------------------------------
    decided = results[results["final_decision"].isin(["correct", "incorrect"])].copy()

    coverage = len(decided) / len(results)
    review_rate = 1 - coverage

    if len(decided) > 0:
        y_true_decided = decided["label_id"].tolist()
        y_pred_decided = decided["final_decision"].map(
            {"incorrect": 0, "correct": 1}
        ).tolist()

        decided_report = classification_report(
            y_true_decided,
            y_pred_decided,
            target_names=["incorrect", "correct"],
            labels=[0, 1],
            digits=2,
            zero_division=0,
        )
    else:
        decided_report = "No decided cases - everything went to review.\n"

    # ------------------------------------------------------------
    # Safety counts for incorrect samples
    # ------------------------------------------------------------
    wrong_total = len(results[results["true_label"] == "incorrect"])

    wrong_rejected = len(
        results[
            (results["true_label"] == "incorrect")
            & (results["final_decision"] == "incorrect")
        ]
    )

    wrong_possible_incorrect = len(
        results[
            (results["true_label"] == "incorrect")
            & (results["final_decision"] == "possible_incorrect")
        ]
    )

    wrong_possible_correct = len(
        results[
            (results["true_label"] == "incorrect")
            & (results["final_decision"] == "possible_correct")
        ]
    )

    wrong_accepted = len(
        results[
            (results["true_label"] == "incorrect")
            & (results["final_decision"] == "correct")
        ]
    )

    # ------------------------------------------------------------
    # Safety counts for correct samples
    # ------------------------------------------------------------
    correct_total = len(results[results["true_label"] == "correct"])

    correct_accepted = len(
        results[
            (results["true_label"] == "correct")
            & (results["final_decision"] == "correct")
        ]
    )

    correct_possible_correct = len(
        results[
            (results["true_label"] == "correct")
            & (results["final_decision"] == "possible_correct")
        ]
    )

    correct_possible_incorrect = len(
        results[
            (results["true_label"] == "correct")
            & (results["final_decision"] == "possible_incorrect")
        ]
    )

    correct_rejected = len(
        results[
            (results["true_label"] == "correct")
            & (results["final_decision"] == "incorrect")
        ]
    )

    # ------------------------------------------------------------
    # Per-class outcome percentages.
    #
    # For each true label, shows what fraction of samples ended up
    # in each final_decision category. possible_* is its own
    # "review" slice - not folded into correct or incorrect.
    # ------------------------------------------------------------
    correct_pct_accepted = (correct_accepted / correct_total * 100) if correct_total > 0 else 0.0
    correct_pct_possible_correct = (correct_possible_correct / correct_total * 100) if correct_total > 0 else 0.0
    correct_pct_possible_incorrect = (correct_possible_incorrect / correct_total * 100) if correct_total > 0 else 0.0
    correct_pct_rejected = (correct_rejected / correct_total * 100) if correct_total > 0 else 0.0

    wrong_pct_rejected = (wrong_rejected / wrong_total * 100) if wrong_total > 0 else 0.0
    wrong_pct_possible_incorrect = (wrong_possible_incorrect / wrong_total * 100) if wrong_total > 0 else 0.0
    wrong_pct_possible_correct = (wrong_possible_correct / wrong_total * 100) if wrong_total > 0 else 0.0
    wrong_pct_accepted = (wrong_accepted / wrong_total * 100) if wrong_total > 0 else 0.0

    # ------------------------------------------------------------
    # Save full reports
    # ------------------------------------------------------------
    report_path = f"{report_folder}/classification_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("PER-CLASS OUTCOME BREAKDOWN\n")
        f.write("===========================\n")
        f.write(f"True label: correct (total={correct_total})\n")
        f.write(f"  correct (auto-accepted):       {correct_accepted:3d}  ({correct_pct_accepted:5.1f}%)\n")
        f.write(f"  possible_correct (review):     {correct_possible_correct:3d}  ({correct_pct_possible_correct:5.1f}%)\n")
        f.write(f"  possible_incorrect (review):   {correct_possible_incorrect:3d}  ({correct_pct_possible_incorrect:5.1f}%)\n")
        f.write(f"  incorrect (auto-rejected):     {correct_rejected:3d}  ({correct_pct_rejected:5.1f}%)\n\n")

        f.write(f"True label: incorrect (total={wrong_total})\n")
        f.write(f"  incorrect (auto-rejected):     {wrong_rejected:3d}  ({wrong_pct_rejected:5.1f}%)\n")
        f.write(f"  possible_incorrect (review):   {wrong_possible_incorrect:3d}  ({wrong_pct_possible_incorrect:5.1f}%)\n")
        f.write(f"  possible_correct (review):     {wrong_possible_correct:3d}  ({wrong_pct_possible_correct:5.1f}%)\n")
        f.write(f"  correct (auto-accepted):       {wrong_accepted:3d}  ({wrong_pct_accepted:5.1f}%)\n\n")

        f.write("SEMANTIC METRICS\n")
        f.write("================\n")
        f.write("possible_correct is counted as correct\n")
        f.write("possible_incorrect is counted as incorrect\n\n")
        f.write(semantic_report)

        f.write("\n\nAUTOMATIC-ACCEPTANCE SAFETY METRICS\n")
        f.write("===================================\n")
        f.write("only final_decision == correct is counted as accepted correct\n")
        f.write("all possible_* decisions are counted as not automatically accepted\n\n")
        f.write(automatic_report)

        f.write("\n\nDECIDED-ONLY METRICS\n")
        f.write("====================\n")
        f.write("possible_correct and possible_incorrect are EXCLUDED entirely\n")
        f.write("(not counted as correct or incorrect - they need manual listening)\n\n")
        f.write(f"coverage (decided / total): {coverage:.2%}\n")
        f.write(f"review rate (possible_* / total): {review_rate:.2%}\n\n")
        f.write(decided_report)

    predictions_path = f"{report_folder}/predictions_{word}.csv"
    results.to_csv(predictions_path, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------
    # Mistakes according to semantic decision
    # ------------------------------------------------------------
    mistakes = results[
        results["true_label"] != results["semantic_binary_prediction"]
    ].copy()

    mistakes_path = f"{report_folder}/mistakes_{word}.csv"
    mistakes.to_csv(mistakes_path, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------
    # Review cases: possible_correct and possible_incorrect
    # ------------------------------------------------------------
    review_cases = results[
        results["final_decision"].isin(
            ["possible_correct", "possible_incorrect"]
        )
    ].copy()

    review_cases_path = f"{report_folder}/review_cases_{word}.csv"
    review_cases.to_csv(review_cases_path, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------
    # Confusion matrix for semantic prediction
    # ------------------------------------------------------------
    cm = confusion_matrix(y_true, y_pred_semantic)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["incorrect", "correct"],
    )

    display.plot()
    confusion_matrix_path = f"{report_folder}/confusion_matrix.png"
    plt.savefig(confusion_matrix_path)
    plt.close()

    # ------------------------------------------------------------
    # One-class SVM evaluation (if model exists)
    # ------------------------------------------------------------
    ocsvm_path = os.path.join(report_folder, "..", "model_oneclass.joblib")
    ocsvm_path = os.path.normpath(ocsvm_path)

    if os.path.exists(ocsvm_path):
        evaluate_one_class_model(
            ocsvm_path=ocsvm_path,
            test_csv=test_csv,
            report_folder=report_folder,
            word=word,
        )
        
    # ------------------------------------------------------------
    # Print final evaluation
    # ------------------------------------------------------------
    print("\n============================================================")
    print("  FINAL EVALUATION")
    print("============================================================\n")

    print(f"Used probability threshold: {best_threshold}")
    print(f"Used phonetic cutoff:       {best_phonetic_cutoff}\n")

    print("PER-CLASS OUTCOME BREAKDOWN")
    print("===========================")
    print(f"True label: correct (total={correct_total})")
    print(f"  correct (auto-accepted):       {correct_accepted:3d}  ({correct_pct_accepted:5.1f}%)")
    print(f"  possible_correct (review):     {correct_possible_correct:3d}  ({correct_pct_possible_correct:5.1f}%)")
    print(f"  possible_incorrect (review):   {correct_possible_incorrect:3d}  ({correct_pct_possible_incorrect:5.1f}%)")
    print(f"  incorrect (auto-rejected):     {correct_rejected:3d}  ({correct_pct_rejected:5.1f}%)")

    print(f"\nTrue label: incorrect (total={wrong_total})")
    print(f"  incorrect (auto-rejected):     {wrong_rejected:3d}  ({wrong_pct_rejected:5.1f}%)")
    print(f"  possible_incorrect (review):   {wrong_possible_incorrect:3d}  ({wrong_pct_possible_incorrect:5.1f}%)")
    print(f"  possible_correct (review):     {wrong_possible_correct:3d}  ({wrong_pct_possible_correct:5.1f}%)")
    print(f"  correct (auto-accepted):       {wrong_accepted:3d}  ({wrong_pct_accepted:5.1f}%)")

    print("\nSEMANTIC METRICS")
    print("================")
    print("possible_correct is counted as correct")
    print("possible_incorrect is counted as incorrect\n")
    print(semantic_report)

    print("\nAUTOMATIC-ACCEPTANCE SAFETY METRICS")
    print("===================================")
    print("only final_decision == correct is counted as accepted correct")
    print("all possible_* decisions are counted as not automatically accepted\n")
    print(automatic_report)

    print("\nDECIDED-ONLY METRICS")
    print("====================")
    print("possible_correct and possible_incorrect are EXCLUDED entirely")
    print("(not counted as correct or incorrect - they need manual listening)\n")
    print(f"coverage (decided / total): {coverage:.2%}")
    print(f"review rate (possible_* / total): {review_rate:.2%}\n")
    print(decided_report)

    print("\nOutput meaning:")
    print("correct              = automatically accepted")
    print("incorrect            = rejected")
    print("possible_correct     = listen manually, probably correct")
    print("possible_incorrect   = listen manually, probably incorrect")

    print(f"\nSaved report:         {report_path}")
    print(f"Saved predictions:    {predictions_path}")
    print(f"Saved mistakes:       {mistakes_path}")
    print(f"Saved review cases:   {review_cases_path}")
    print(f"Saved confusion mat.: {confusion_matrix_path}")
    print(f"Number of semantic mistakes: {len(mistakes)}")
    print(f"Number of review cases:      {len(review_cases)}")

    # ------------------------------------------------------------
    # Print semantic mistake summary
    # ------------------------------------------------------------
    if len(mistakes) > 0:
        columns_to_show = [
            "true_label",
            "model_prediction",
            "final_decision",
            "semantic_binary_prediction",
            "decision_reason",
            "phonetic_score",
            "probability_correct",
            "matched_transcription",
            "whisper_transcription",
            "whisper_similarity",
            "detected_status",
            "transcription_exact_match",
            "review_required",
        ]

        existing_columns = [
            col for col in columns_to_show
            if col in mistakes.columns
        ]

        print("\nSemantic mistake summary:")
        print(mistakes[existing_columns])

    # ------------------------------------------------------------
    # Print review summary
    # ------------------------------------------------------------
    if len(review_cases) > 0:
        columns_to_show = [
            "true_label",
            "model_prediction",
            "final_decision",
            "semantic_binary_prediction",
            "decision_reason",
            "phonetic_score",
            "probability_correct",
            "matched_transcription",
            "detected_status",
            "transcription_exact_match",
            "review_required",
        ]

        existing_columns = [
            col for col in columns_to_show
            if col in review_cases.columns
        ]


        print("\nReview cases summary:")
        print(review_cases[existing_columns])




######################## ONE CLASS ###############################

def evaluate_one_class_model(ocsvm_path, test_csv, report_folder, word):
    """
    Evaluate the one-class SVM on the test set.

    Reports:
        1. How many correct samples were predicted as correct (inliers)
        2. How many incorrect samples were predicted as not-correct (outliers)
    """

    import joblib
    import os

    if not os.path.exists(ocsvm_path):
        print(f"⚠️  No one-class model found at {ocsvm_path}, skipping.")
        return

    ocsvm_data = joblib.load(ocsvm_path)
    ocsvm = ocsvm_data["model"]
    feature_columns = ocsvm_data["feature_columns"]
    best_nu = ocsvm_data["best_nu"]

    test_df = pd.read_csv(test_csv)

    X_test = test_df[feature_columns]

    # +1 = inlier (predicted correct), -1 = outlier (predicted wrong)
    preds = ocsvm.predict(X_test)

    # Signed distance from decision boundary:
    # Positive = confidently inside correct boundary
    # Negative = outside correct boundary (likely wrong)
    scores = ocsvm.decision_function(X_test)

    correct_mask = test_df["label"] == "correct"
    incorrect_mask = test_df["label"] == "incorrect"

    correct_total = correct_mask.sum()
    incorrect_total = incorrect_mask.sum()

    correct_predicted_correct = (preds[correct_mask] == 1).sum()
    incorrect_predicted_wrong = (preds[incorrect_mask] == -1).sum()

    correct_pct = correct_predicted_correct / correct_total * 100 if correct_total > 0 else 0
    incorrect_pct = incorrect_predicted_wrong / incorrect_total * 100 if incorrect_total > 0 else 0

    print(f"\n{'=' * 60}")
    print("  ONE-CLASS SVM EVALUATION")
    print(f"  (trained on correct samples only)")
    print(f"{'=' * 60}\n")
    print(f"  nu used: {best_nu}\n")
    print(f"  True correct  → predicted correct:  {correct_predicted_correct}/{correct_total} ({correct_pct:.1f}%)")
    print(f"  True incorrect → predicted wrong:   {incorrect_predicted_wrong}/{incorrect_total} ({incorrect_pct:.1f}%)")

    # ----------------------------------------------------------------
    # Save one-class scores CSV — contains per-sample scores for
    # threshold analysis without needing to re-load the joblib.
    # ----------------------------------------------------------------
    scores_df = test_df[["filename", "label", "phonetic_score"]].copy()
    scores_df["oneclass_pred"] = preds           # +1 correct, -1 wrong
    scores_df["oneclass_score"] = scores.round(6) # signed distance from boundary

    # Normalize to 0-100 for readability
    s_min = scores.min()
    s_max = scores.max()
    if s_max > s_min:
        scores_df["oneclass_score_norm"] = ((scores - s_min) / (s_max - s_min) * 100).round(1)
    else:
        scores_df["oneclass_score_norm"] = 50.0

    scores_path = os.path.join(report_folder, f"oneclass_scores_{word}.csv")
    scores_df.to_csv(scores_path, index=False, encoding="utf-8-sig")

    # Save text report
    report_path = os.path.join(report_folder, f"oneclass_report_{word}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("ONE-CLASS SVM EVALUATION\n")
        f.write("(trained on correct samples only)\n\n")
        f.write(f"nu used: {best_nu}\n\n")
        f.write(f"True correct  → predicted correct:  {correct_predicted_correct}/{correct_total} ({correct_pct:.1f}%)\n")
        f.write(f"True incorrect → predicted wrong:   {incorrect_predicted_wrong}/{incorrect_total} ({incorrect_pct:.1f}%)\n")

    print(f"\n  Saved report: {report_path}")
    print(f"  Saved scores: {scores_path}")