# wav2vec2_features.py

import torch
import numpy as np

import globalVars as gv
from globalVars import SAMPLE_RATE


def load_wav2vec2(model_name):
    """
    Load the wav2vec2 model and processor.

    We do not train wav2vec2 — only use it to extract embeddings.
    eval() disables dropout and gradient computation.

    Returns:
        processor: prepares audio arrays for the model
        model:     produces hidden state representations
    """

    print(f"Loading wav2vec2 model ({model_name})...")

    from transformers import Wav2Vec2Processor, Wav2Vec2Model

    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name)
    model.eval()

    return processor, model


def extract_wav2vec2_embedding(audio, processor, model):
    """
    Extract a fixed-size embedding from a preprocessed audio array.

    Steps:
        1. Prepare audio for wav2vec2
        2. Extract hidden states from the chosen layer
        3. Mean + std pooling across time frames
        4. Concatenate → one vector per audio clip

    Why mean + std pooling:
        Mean captures the average acoustic content.
        Std captures how much the speech varies over time.
        Together they give the classifier more information
        than mean alone.

    The layer is controlled by gv.EMBEDDING_LAYER at runtime.
    We read it via gv.EMBEDDING_LAYER (not a direct import) so that
    main.py can temporarily override it per word without the change
    being ignored due to Python's import-time value copying.

    Middle layers (6-12) retain more acoustic/pronunciation detail
    than the last layer which is optimized for ASR word identity.

    Input:
        audio: normalized 1D numpy array at SAMPLE_RATE

    Returns:
        embedding: 1D numpy array (2 * hidden_size,)
    """

    inputs = processor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Read EMBEDDING_LAYER at call time via gv — not at import time.
    # This ensures main.py's temporary override takes effect.
    hidden = outputs.hidden_states[gv.EMBEDDING_LAYER].squeeze(0)

    mean_vector = hidden.mean(dim=0)
    std_vector = hidden.std(dim=0, unbiased=False)

    embedding = torch.cat([mean_vector, std_vector], dim=0).numpy()
    return embedding



def extract_wav2vec2_embedding_multilayer(audio, processor, model, layers):
    """
    Extract and concatenate embeddings from multiple layers.

    Each layer contributes mean+std pooling → 2048 features.
    Total output: len(layers) * 2048 features.

    Why multiple layers:
        Different layers capture different aspects — lower layers
        retain more acoustic/phonetic detail, higher layers more
        linguistic/word-identity content. Concatenating gives the
        SVC access to both simultaneously.
    """
    inputs = processor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    embeddings = []
    for layer in layers:
        hidden = outputs.hidden_states[layer].squeeze(0)
        mean_vector = hidden.mean(dim=0)
        std_vector = hidden.std(dim=0, unbiased=False)
        embeddings.append(torch.cat([mean_vector, std_vector], dim=0).numpy())

    return np.concatenate(embeddings)