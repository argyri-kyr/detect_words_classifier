# decision_rules.py

import pandas as pd

from load_phonetic_scores import normalize_greek_phonetic
from globalVars import AUTO_ACCEPT_CONFIDENCE, WHISPER_ACCEPT_THRESHOLD, ONECLASS_GATE_PARAMS


def value_is_missing(value):
    """
    Returns True when a value is missing, empty, or NaN.
    """

    if value is None:
        return True

    if pd.isna(value):
        return True

    if str(value).strip() == "":
        return True

    return False


def normalize_for_transcription_match(text):
    """
    Normalizes a transcription for comparison against the target word.

    Used for both matched_transcription (Excel/forced aligner) and
    whisper_transcription (free Whisper). Same normalization applies
    to both because Whisper can also split words or produce minor
    orthographic variants just like the forced aligner.

    Examples:
        "στα φιδιασμένα"  -> "σταφιδιασμένα"
        "στα αφιδιασμένα" -> "στααφιδιασμένα" -> "σταφιδιασμένα"
    """

    if value_is_missing(text):
        return ""

    normalized = normalize_greek_phonetic(str(text))
    normalized = normalized.replace(" ", "")
    normalized = normalized.replace("-", "")
    normalized = normalized.replace("_", "")

    if normalized.startswith("στααφ"):
        normalized = "σταφ" + normalized[len("στααφ"):]

    return normalized


def transcription_matches_target(transcription, target_word):
    """
    Checks whether a transcription matches the target word.

    Intentionally tolerant — handles accents, phonetic variants,
    spaces inside words, and the common "στα αφ..." split error.
    """

    if value_is_missing(transcription) or value_is_missing(target_word):
        return False

    return (
        normalize_for_transcription_match(transcription)
        == normalize_for_transcription_match(target_word)
    )


def make_base_model_decision(
    probability_correct,
    phonetic_score,
    probability_threshold,
    phonetic_cutoff,
):
    """
    Binary decision logic based on model probability and phonetic cutoff.

    When model is highly confident, skip the phonetic cutoff to avoid
    over-rejection of borderline correct pronunciations.
    """

    if phonetic_cutoff is not None:
        if pd.notna(phonetic_score) and phonetic_score < phonetic_cutoff:
            if probability_correct >= AUTO_ACCEPT_CONFIDENCE:
                return "correct", "model_confident_overrides_phonetic_cutoff"
            return "incorrect", "phonetic_cutoff_forced_incorrect"

    if probability_correct >= probability_threshold:
        return "correct", "model_probability_correct"

    return "incorrect", "model_probability_incorrect"


def make_final_decision(
    target_word,
    probability_correct,
    phonetic_score,
    probability_threshold,
    phonetic_cutoff,
    matched_transcription=None,
    detected_status=None,
    whisper_similarity=None,
    whisper_transcription=None,
    oneclass_score=None,
):
    """
    Combined decision rule using binary SVC + optional one-class SVM gate.

    Final outputs:
        correct            = automatically accepted
        incorrect          = rejected
        possible_correct   = listen manually, probably correct
        possible_incorrect = listen manually, probably incorrect

    One-class gate logic:
        Applied only for words listed in ONECLASS_GATE_PARAMS.
        Gate fires when BOTH are true:
            1. probability_correct < prob_threshold (SVC not overwhelmingly sure)
            2. oneclass_score < oc_threshold (one-class SVM is skeptical)
        When gate fires, auto-accept becomes possible_correct (review).

        Thresholds are empirically determined per word from score
        distribution analysis — not a fixed global value.

    Transcription match:
        Uses matched_transcription (Excel/forced aligner).
        Whisper transcription is NOT used for matching — too strict
        for child speech variations.
        Whisper similarity is kept as safety gate only (currently disabled).
    """

    model_prediction, model_reason = make_base_model_decision(
        probability_correct=probability_correct,
        phonetic_score=phonetic_score,
        probability_threshold=probability_threshold,
        phonetic_cutoff=phonetic_cutoff,
    )

    transcription_match = transcription_matches_target(
        transcription=matched_transcription,
        target_word=target_word,
    )
    transcription_available = not value_is_missing(matched_transcription)

    status_text = ""
    if not value_is_missing(detected_status):
        status_text = str(detected_status).strip().lower()
    status_found = status_text == "found"

    # Whisper gate (currently disabled since USE_WHISPER_SIMILARITY=False)
    whisper_available = (
        whisper_similarity is not None
        and not value_is_missing(whisper_similarity)
        and WHISPER_ACCEPT_THRESHOLD is not None
    )
    whisper_disagrees = (
        whisper_available
        and float(whisper_similarity) < WHISPER_ACCEPT_THRESHOLD
    )

    # One-class gate — per-word empirical thresholds.
    # Fires when binary SVC is not overwhelmingly certain AND
    # one-class SVM score is below the word-specific threshold.
    # Only active for words listed in ONECLASS_GATE_PARAMS.
    oneclass_gate_fires = False
    if (
        oneclass_score is not None
        and not value_is_missing(oneclass_score)
        and target_word in ONECLASS_GATE_PARAMS
    ):
        gate_params = ONECLASS_GATE_PARAMS[target_word]
        oneclass_gate_fires = (
            float(probability_correct) < gate_params["prob_threshold"]
            and float(oneclass_score) < gate_params["oc_threshold"]
        )
    
    # ------------------------------------------------------------
    # Rule 0: Very low probability — always possible_incorrect,
    # regardless of other evidence (phonetic score, transcription).
    # Catches cases where the model is strongly skeptical even if
    # other signals are superficially supportive.
    # ------------------------------------------------------------
    VERY_LOW_PROB_THRESHOLD = 0.20
    if probability_correct < VERY_LOW_PROB_THRESHOLD:
        return {
            "model_prediction": model_prediction,
            "final_decision": "possible_incorrect",
            "decision_reason": "very_low_model_probability",
            "transcription_exact_match": transcription_matches_target(
                matched_transcription, target_word
            ),
            "review_required": True,
        }
        
    # ------------------------------------------------------------
    # Rule 1: Very confident correct.
    # probability ≥ 0.90 AND phonetic ≥ 95
    # Note: one-class gate can still fire here if prob < prob_threshold
    # (prob_threshold is 0.99 for gate words, so 0.90-0.99 range is gated)
    # ------------------------------------------------------------
    if (
        probability_correct >= 0.90
        and pd.notna(phonetic_score)
        and phonetic_score >= 95
    ):
        if whisper_disagrees:
            return {
                "model_prediction": model_prediction,
                "final_decision": "possible_incorrect",
                "decision_reason": "very_confident_but_whisper_disagrees",
                "transcription_exact_match": transcription_match,
                "review_required": True,
            }
        if oneclass_gate_fires:
            return {
                "model_prediction": model_prediction,
                "final_decision": "possible_correct",
                "decision_reason": "oneclass_svm_gate_fired",
                "transcription_exact_match": transcription_match,
                "review_required": True,
            }
        return {
            "model_prediction": model_prediction,
            "final_decision": "correct",
            "decision_reason": "very_confident_model_and_good_phonetic_score",
            "transcription_exact_match": transcription_match,
            "review_required": False,
        }

    # ------------------------------------------------------------
    # Rule 2: Normal automatic correct.
    # model=correct AND transcription matches AND status=Found
    # ------------------------------------------------------------
    if model_prediction == "correct":
        if status_found and transcription_match:
            if whisper_disagrees:
                return {
                    "model_prediction": model_prediction,
                    "final_decision": "possible_incorrect",
                    "decision_reason": "transcription_agrees_but_whisper_disagrees",
                    "transcription_exact_match": True,
                    "review_required": True,
                }
            if oneclass_gate_fires:
                return {
                    "model_prediction": model_prediction,
                    "final_decision": "possible_correct",
                    "decision_reason": "oneclass_svm_gate_fired",
                    "transcription_exact_match": True,
                    "review_required": True,
                }
            return {
                "model_prediction": model_prediction,
                "final_decision": "correct",
                "decision_reason": "model_and_transcription_agree_correct",
                "transcription_exact_match": True,
                "review_required": False,
            }

        # --------------------------------------------------------
        # Rule 3: Model confident, transcription disagrees.
        # --------------------------------------------------------
        if probability_correct >= AUTO_ACCEPT_CONFIDENCE:
            if whisper_disagrees:
                return {
                    "model_prediction": model_prediction,
                    "final_decision": "possible_incorrect",
                    "decision_reason": "model_confident_but_whisper_disagrees",
                    "transcription_exact_match": transcription_match,
                    "review_required": True,
                }
            if oneclass_gate_fires:
                return {
                    "model_prediction": model_prediction,
                    "final_decision": "possible_correct",
                    "decision_reason": "oneclass_svm_gate_fired",
                    "transcription_exact_match": transcription_match,
                    "review_required": True,
                }
            return {
                "model_prediction": model_prediction,
                "final_decision": "correct",
                "decision_reason": "model_confident_despite_transcription_mismatch",
                "transcription_exact_match": transcription_match,
                "review_required": False,
            }

        if not transcription_available:
            reason = "model_correct_but_no_transcription_available"
        elif not status_found:
            reason = "model_correct_but_detected_status_not_found"
        else:
            reason = "model_correct_but_transcription_mismatch"

        return {
            "model_prediction": model_prediction,
            "final_decision": "possible_incorrect",
            "decision_reason": reason,
            "transcription_exact_match": transcription_match,
            "review_required": True,
        }

    # ------------------------------------------------------------
    # Rule 4: Possible correct.
    # model=incorrect but evidence is strong.
    # ------------------------------------------------------------
    if (
        model_prediction == "incorrect"
        and pd.notna(phonetic_score)
        and phonetic_score >= 95
        and probability_correct >= 0.45
        and status_found
        and transcription_match
    ):
        return {
            "model_prediction": model_prediction,
            "final_decision": "possible_correct",
            "decision_reason": "model_incorrect_but_transcription_matches_and_evidence_is_strong",
            "transcription_exact_match": True,
            "review_required": True,
        }

    # ------------------------------------------------------------
    # Rule 5: Everything else is incorrect.
    # ------------------------------------------------------------
    return {
        "model_prediction": model_prediction,
        "final_decision": "incorrect",
        "decision_reason": model_reason,
        "transcription_exact_match": transcription_match,
        "review_required": False,
    }