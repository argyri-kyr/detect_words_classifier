# augment_audio.py
#
# Generates augmented versions of wrong pronunciation clips.
# Augmented files are saved alongside originals in the {word}_wrong/ folder.
# build_metadata() picks them up automatically on the next run.
# To undo: delete all files containing '_aug_' in their filename.

import librosa
import numpy as np
import soundfile as sf
import os

from globalVars import SAMPLE_RATE


def augment_audio(audio, sr=SAMPLE_RATE):
    """
    Generates 5 augmented versions of one audio clip.

    Augmentations applied:
        pitch_up:     shift pitch up 1.5 semitones
        pitch_down:   shift pitch down 1.5 semitones
        stretch_fast: speed up by 10% (shorter duration)
        stretch_slow: slow down by 10% (longer duration)
        noise:        add small Gaussian noise (std=0.005)

    Why these augmentations:
        They preserve the pronunciation error (the "wrongness") while
        creating acoustic variety — simulating different children, different
        microphones, different reading speeds. This helps the SVC learn a
        more robust boundary around the incorrect class.

    Returns:
        List of (augmented_audio, suffix) tuples.
    """

    augmented = []

    # Pitch shift up
    shifted_up = librosa.effects.pitch_shift(audio, sr=sr, n_steps=1.5)
    augmented.append((shifted_up, "pitch_up"))

    # Pitch shift down
    shifted_down = librosa.effects.pitch_shift(audio, sr=sr, n_steps=-1.5)
    augmented.append((shifted_down, "pitch_down"))

    # Time stretch faster
    stretched_fast = librosa.effects.time_stretch(audio, rate=1.1)
    augmented.append((stretched_fast, "stretch_fast"))

    # Time stretch slower
    stretched_slow = librosa.effects.time_stretch(audio, rate=0.9)
    augmented.append((stretched_slow, "stretch_slow"))

    # Gaussian noise
    noise = np.random.normal(0, 0.005, len(audio))
    noisy = np.clip(audio + noise, -1.0, 1.0)
    noisy = noisy / np.max(np.abs(noisy))
    augmented.append((noisy, "noise"))

    return augmented


def augment_wrong_folder(word, word_folder):
    """
    Augments all original .wav files in {word_folder}/{word}_wrong/.

    Skips files already augmented (filename contains '_aug_') to avoid
    augmenting augmented files on repeated runs.

    Args:
        word:        the Greek word string (e.g. 'δραστηριότητες')
        word_folder: full path to the word's data folder
    """

    wrong_folder = os.path.join(word_folder, f"{word}_wrong")

    if not os.path.exists(wrong_folder):
        print(f"  No wrong folder found: {wrong_folder}")
        return 0

    # Only augment original files — skip any already-augmented ones
    wav_files = [
        f for f in os.listdir(wrong_folder)
        if f.endswith('.wav') and '_aug_' not in f
    ]

    if len(wav_files) == 0:
        print(f"  No original wrong clips found in: {wrong_folder}")
        return 0

    print(f"  Augmenting {len(wav_files)} original clips -> {len(wav_files) * 5} new clips")

    for wav_file in wav_files:
        audio, _ = librosa.load(
            os.path.join(wrong_folder, wav_file),
            sr=SAMPLE_RATE
        )
        augmented = augment_audio(audio)

        base = os.path.splitext(wav_file)[0]
        for aug_audio, suffix in augmented:
            out_name = f"{base}_aug_{suffix}.wav"
            out_path = os.path.join(wrong_folder, out_name)
            sf.write(out_path, aug_audio, SAMPLE_RATE)

    print(f"  Saved to: {wrong_folder}")
    return len(wav_files) * 5

