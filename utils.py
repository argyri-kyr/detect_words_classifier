from pathlib import Path
import pandas as pd


def create_folder(folder_path):
    """
    Create a folder if it does not exist.

    Example:
        create_folder("outputs")
    """

    folder_path = Path(folder_path)
    folder_path.mkdir(parents=True, exist_ok=True)
    return folder_path


def read_metadata(csv_path):
    """
    Read the metadata CSV file.

    The CSV must have these columns:
        filepath,target_word,label

    Example:
        data/audio1.wav,καλημέρα,correct
        data/audio2.wav,καλημέρα,incorrect
    """

    df = pd.read_csv(csv_path)

    required_columns = ["filepath", "target_word", "label"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing column in CSV: {column}")

    # Clean the label values.
    # Example: " Correct " becomes "correct"
    df["label"] = df["label"].str.lower().str.strip()

    return df


def save_csv(df, output_path):
    """
    Save a pandas DataFrame as CSV.
    """

    output_path = Path(output_path)

    # Create the parent folder if needed.
    create_folder(output_path.parent)

    df.to_csv(output_path, index=False)
    

def ensure_dir(folder_path):
    """
    Compatibility alias for create_folder().
    Used by preprocess_audio.py.
    """
    return create_folder(folder_path)


def load_metadata(csv_path):
    """
    Compatibility alias for read_metadata().
    Used by preprocess_audio.py.
    """
    return read_metadata(csv_path)


def save_dataframe(df, output_path):
    """
    Compatibility alias for save_csv().
    Used by preprocess_audio.py.
    """
    return save_csv(df, output_path)