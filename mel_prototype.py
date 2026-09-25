

import numpy as np
import librosa
from dtaidistance import dtw_ndim
from tqdm import tqdm


from globalVars import SAMPLE_RATE


# Mel spectrogram parameters.
# 40 mel bands, 10ms hop, 25ms window — standard for speech.
N_MELS = 40
HOP_LENGTH = 160   # 10ms at 16kHz
N_FFT = 400        # 25ms window at 16kHz


def compute_mel_spectrogram(audio):
    """
    Compute a mel-spectrogram sequence from a normalized audio array.

    Returns a 2D array of shape (n_frames, n_mels).
    Each row is one time frame, each column is one mel frequency band.

    Why mel-spectrogram:
        It captures the frequency content over time — the waveform shape
        you described. The presence/absence of phonemes like ρ creates
        distinctive frequency patterns at specific time positions.
        Two children saying αρδευτικός and αδευτικός in the same duration
        will have different mel shapes where the ρ should appear.

    Why log scale (dB):
        Human hearing is logarithmic — perceptually meaningful differences
        are better captured in dB than in raw power.

    Why transpose to (n_frames, n_mels):
        DTW operates on sequences of vectors, one vector per time step.
    """

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.T


def compute_medoid(sequences):
    """
    Find the medoid of a list of mel-spectrogram sequences.

    The medoid is the sequence with the smallest total DTW distance
    to all other sequences. It is a real sample from the dataset —
    not an average — so it avoids the problem of averaging
    variable-length sequences.

    Why medoid instead of mean:
        Children read at different speeds so sequences have different
        lengths. You cannot directly average variable-length sequences.
        The medoid sidesteps this by picking the most central real sample.

    Why recompute on every training run:
        As new correct samples are added to the dataset, the medoid
        should reflect the current distribution of correct pronunciations.

    Input:
        sequences: list of 2D numpy arrays (n_frames, n_mels)

    Returns:
        medoid:       the most central sequence (a real sample)
        medoid_index: its index in the input list
    """

    n = len(sequences)

    if n == 1:
        return sequences[0], 0

    # Total number of pairwise comparisons: n*(n-1)/2
    total_pairs = n * (n - 1) // 2

    total_distances = np.zeros(n)

    with tqdm(total=total_pairs, desc="Computing medoid distances") as pbar:
        for i in range(n):
            for j in range(i + 1, n):
                dist = dtw_ndim.distance(sequences[i], sequences[j])
                total_distances[i] += dist
                total_distances[j] += dist
                pbar.update(1)

    medoid_index = int(np.argmin(total_distances))
    return sequences[medoid_index], medoid_index


def compute_dtw_distance_to_medoid(mel_sequence, medoid_sequence):
    """
    Compute DTW distance between one mel-spectrogram sequence
    and the correct pronunciation prototype (medoid).

    A small distance means the sample sounds similar to a correct
    pronunciation. A large distance means it sounds different —
    possibly because a phoneme is missing or substituted.

    Returns:
        distance: float (non-negative, 0 = identical to medoid)
    """

    from dtaidistance import dtw_ndim

    dist = dtw_ndim.distance(mel_sequence, medoid_sequence)
    return float(dist)