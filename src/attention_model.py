"""Attention-based sentiment classification model (TensorFlow/Keras).

Architecture
------------
    Input (token ids, shape: max_length)
      -> Embedding (vocab_size x embedding_dim, masking the PAD token)
      -> Bidirectional LSTM (returns full sequence)
      -> Attention layer (Bahdanau-style additive attention)
      -> Dense(softmax) -> [negative, neutral, positive]

The attention layer produces a weight for every input token; those weights are
what the visualization module uses to highlight influential words.

NOTE: this module defines the architecture only. No trained weights are
bundled with the repository - a model must be produced by ``training/train.py``
before inference (src/predictor.py) will run.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

ATTENTION_LAYER_NAME = "token_attention"


class AttentionLayer(layers.Layer):
    """Bahdanau-style additive attention over the full token sequence.

    Output: (batch, timesteps) - the attention probability distribution over
    tokens. The main model combines it with the sequence via a Dot product
    (see :func:`build_model`), and :func:`build_attention_probe` exposes the
    same distribution as a model output for inference-time visualisation.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        features = int(input_shape[-1])
        self.W = self.add_weight(
            name="attention_kernel",
            shape=(features, features),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="attention_bias",
            shape=(features,),
            initializer="zeros",
            trainable=True,
        )
        self.u = self.add_weight(
            name="attention_context",
            shape=(features,),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs, mask=None):
        # inputs: (batch, time, features) -> output: (batch, time)
        score = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        score = tf.tensordot(score, self.u, axes=1)  # (batch, time)
        if mask is not None:
            score += (1.0 - tf.cast(mask, score.dtype)) * -1e9
        return tf.nn.softmax(score, axis=1)

    def compute_mask(self, inputs, mask=None):
        # The attention distribution needs no padding mask downstream.
        return None

    def get_config(self):
        config = super().get_config()
        return config


def build_model(
    vocab_size: int,
    max_length: int,
    *,
    embedding_dim: int = 128,
    lstm_units: int = 64,
    num_classes: int = 3,
    learning_rate: float = 1e-3,
    name: str = "sentiment_attention",
) -> keras.Model:
    """Build and compile the Embedding -> BiLSTM -> Attention -> Dense model.

    Parameters
    ----------
    vocab_size:
        Size of the vocabulary produced by ``src.preprocessing.build_vocab``
        (includes the <PAD>/<UNK> slots).
    max_length:
        Fixed padded sequence length used by training and inference.
    num_classes:
        Number of sentiment classes (default 3: negative/neutral/positive).
    """
    inputs = keras.Input(shape=(max_length,), name="token_ids")
    embedded = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        mask_zero=True,
        name="embedding",
    )(inputs)
    sequence = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True), name="bilstm"
    )(embedded)
    # attention distribution over tokens: (batch, time)
    weights = AttentionLayer(name=ATTENTION_LAYER_NAME)(sequence)
    # sum_t weight_t * hidden_t  ->  (batch, features)
    x = layers.Dot(axes=1, name="attention_weighted_sum")([weights, sequence])
    x = layers.Dropout(0.3, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="sentiment")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=name)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_attention_probe(model: keras.Model) -> keras.Model:
    """Return a model that outputs ``(class_probabilities, attention_weights)``.

    Reuses the trained layers of ``model`` so the attention distribution over
    tokens is available at inference time without retraining anything.
    """
    attention_output = model.get_layer(ATTENTION_LAYER_NAME).output
    probe = keras.Model(
        inputs=model.input,
        outputs=[model.output, attention_output],
        name=model.name + "_probe",
    )
    return probe


def custom_objects() -> dict:
    """Custom objects needed by ``keras.models.load_model``."""
    return {AttentionLayer.__name__: AttentionLayer}
