# Greek Pronunciation Correct/Incorrect Classifier

This project trains a binary classifier for Greek word pronunciation audio.

Input:
- Audio file
- Target Greek word

Output:
- correct
- incorrect

## Dataset format

Create a CSV file like this:

```csv
filepath,target_word,label,speaker_id
data/raw/audio001.wav,καλημέρα,correct,spk01
data/raw/audio002.wav,καλημέρα,incorrect,spk02
data/raw/audio003.wav,ευχαριστώ,correct,spk01
```

`speaker_id` is optional but recommended. If present, evaluation can split by speaker.

## Install

```bash
pip install -r requirements.txt
```

You also need `ffmpeg` installed for preprocessing.

## Create metadata automatically from folders

If your files are arranged as:

```text
dataset/
  correct/
  incorrect/
```

run:

```bash
python make_metadata_from_folders.py --dataset-dir dataset --output data/metadata.csv
```

By default, the script assumes the target word is the part of the filename before the first underscore.

Example:

```text
καλημέρα_001.wav -> target_word = καλημέρα
```

## Full pipeline

```bash
python main.py \
  --metadata data/metadata.csv \
  --processed-dir data/processed \
  --features-out outputs/features.csv \
  --model-out outputs/model.joblib \
  --report-dir outputs/report
```

## Predict one audio file

```bash
python predict.py \
  --model outputs/model.joblib \
  --audio data/processed/example.wav \
  --target-word καλημέρα
```
