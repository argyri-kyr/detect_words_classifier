

import os
from pathlib import Path
import argparse
import pandas as pd

import globalVars as gv
from utils import save_csv


def get_wav_paths(word, label):
    """
    Find all .wav files for one word and one label.

    label can be:
        "correct"
        "wrong"

    Example path:
        folder1/folder2/word/word_correct
    """

    folder = os.path.join(
        gv.folder1,
        gv.folder2,
        word,
        f"{word}_{label}"
    )

    if not os.path.exists(folder):
        print(f"Warning: folder does not exist: {folder}")
        return []

    paths = []

    for filename in os.listdir(folder):
        if filename.lower().endswith(".wav"):
            full_path = os.path.join(folder, filename)
            paths.append(full_path)

    paths = sorted(paths)

    if len(paths) == 0:
        print(f"Warning: no wav files found in: {folder}")

    return paths


def discover_words():
    """
    Automatically find all word folders inside folder1/folder2.

    Example:
        detect_words_decodewordspseudo_data/
          καλημέρα/
          νερό/
          σπίτι/

    This function returns:
        ["καλημέρα", "νερό", "σπίτι"]
    """

    base_folder = Path(gv.folder1) / gv.folder2

    if not base_folder.exists():
        raise FileNotFoundError(f"Base folder not found: {base_folder}")

    words = []

    for item in sorted(base_folder.iterdir()):
        if item.is_dir():
            words.append(item.name)

    return words


def build_metadata(words, output_csv):
    """
    Create the metadata CSV.

    The output CSV will have:
        filepath,target_word,label

    Important:
        Your folder name is "wrong"
        but our model label will be "incorrect"
    """

    rows = []

    for word in words:
        # Correct audios
        correct_paths = get_wav_paths(word, "correct")

        for path in correct_paths:
            rows.append({
                "filepath": path,
                "target_word": word,
                "label": "correct",
            })

        # Wrong audios.
        # We rename "wrong" to "incorrect" for the model.
        wrong_paths = get_wav_paths(word, "wrong")

        for path in wrong_paths:
            rows.append({
                "filepath": path,
                "target_word": word,
                "label": "incorrect",
            })

    df = pd.DataFrame(rows)

    if len(df) == 0:
        raise RuntimeError("No audio files found. Check your folders and word names.")

    save_csv(df, output_csv)

    print(f"Saved metadata to: {output_csv}")
    print()
    print("Number of files per label:")
    print(df["label"].value_counts())

    return df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        required=True,
        help="Where to save metadata CSV, for example data/metadata.csv"
    )

    parser.add_argument(
        "--words",
        nargs="*",
        default=None,
        help="Optional list of words. If empty, words are discovered automatically."
    )

    args = parser.parse_args()

    # If user gives words, use them.
    # Otherwise discover words automatically.
    if args.words is None or len(args.words) == 0:
        words = discover_words()
    else:
        words = args.words

    build_metadata(words, args.output)


if __name__ == "__main__":
    main()