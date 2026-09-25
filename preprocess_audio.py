from pathlib import Path
import argparse
import subprocess
from tqdm import tqdm

from globalVars import SAMPLE_RATE
from utils import load_metadata, ensure_dir, save_dataframe


def convert_audio_ffmpeg(input_path, output_path, sample_rate=SAMPLE_RATE):
    input_path = Path(input_path)
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-ac", "1",
        "-ar", str(sample_rate),
        "-vn",
        str(output_path),
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {input_path}\\n\\n{result.stderr}"
        )


def preprocess_dataset(metadata_csv, processed_dir, output_metadata_csv=None):
    df = load_metadata(metadata_csv)
    processed_dir = ensure_dir(processed_dir)

    new_paths = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing audio"):
        input_path = Path(row["filepath"])

        if not input_path.exists():
            raise FileNotFoundError(f"Audio file not found: {input_path}")

        label = row["label"]
        target_word = row["target_word"]

        safe_word = (
            target_word
            .replace("/", "_")
            .replace("\\\\", "_")
            .replace(" ", "_")
        )

        output_name = f"{idx:06d}_{safe_word}_{label}.wav"
        output_path = processed_dir / label / output_name

        convert_audio_ffmpeg(input_path, output_path)
        new_paths.append(str(output_path))

    df["filepath"] = new_paths

    if output_metadata_csv is not None:
        save_dataframe(df, output_metadata_csv)

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, help="Input metadata CSV")
    parser.add_argument("--processed-dir", required=True, help="Directory for processed wav files")
    parser.add_argument("--output-metadata", required=True, help="Output CSV with processed file paths")
    args = parser.parse_args()

    preprocess_dataset(
        metadata_csv=args.metadata,
        processed_dir=args.processed_dir,
        output_metadata_csv=args.output_metadata,
    )


if __name__ == "__main__":
    main()
