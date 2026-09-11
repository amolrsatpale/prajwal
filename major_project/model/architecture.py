"""
Deep Learning Model Architecture for Phishing URL Detection
Hybrid CNN-BiLSTM with statistical feature fusion.

Optimization techniques used:
1. Adam optimizer with configurable learning rate
2. Learning rate scheduling (ReduceLROnPlateau)
3. Dropout regularization
4. Batch normalization
5. L2 kernel regularization
6. Early stopping
7. Model checkpointing (best weights)
8. Gradient clipping
9. Class-weight balancing support
10. Spatial dropout for embeddings
"""

import os

# Suppress TF warnings before import
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks

from .feature_extractor import VOCAB_SIZE, MAX_URL_LEN, NUM_FEATURES


def build_model(
    embedding_dim: int = 64,
    cnn_filters: int = 128,
    cnn_kernel: int = 3,
    lstm_units: int = 64,
    dense_units: int = 128,
    dropout_rate: float = 0.4,
    l2_strength: float = 1e-4,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """
    Build the hybrid CNN-BiLSTM phishing detection model.

    Architecture:
        Branch A – Character-level sequence:
            Embedding → SpatialDropout1D → Conv1D → BatchNorm → MaxPool
            → BiLSTM → Dropout

        Branch B – Statistical features:
            Dense → BatchNorm → Dropout → Dense → BatchNorm → Dropout

        Merge → Dense → BatchNorm → Dropout → Sigmoid output
    """
    l2 = regularizers.l2(l2_strength)

    # ------------------------------------------------------------------
    # Branch A: Character-level sequence input
    # ------------------------------------------------------------------
    char_input = keras.Input(shape=(MAX_URL_LEN,), dtype="int32", name="char_input")

    x = layers.Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=embedding_dim,
        input_length=MAX_URL_LEN,
        name="char_embedding",
    )(char_input)
    x = layers.SpatialDropout1D(0.2, name="spatial_dropout")(x)

    # CNN block
    x = layers.Conv1D(
        cnn_filters, cnn_kernel, activation="relu",
        padding="same", kernel_regularizer=l2, name="conv1d_1"
    )(x)
    x = layers.BatchNormalization(name="bn_conv1")(x)
    x = layers.MaxPooling1D(pool_size=2, name="maxpool_1")(x)

    x = layers.Conv1D(
        cnn_filters // 2, cnn_kernel, activation="relu",
        padding="same", kernel_regularizer=l2, name="conv1d_2"
    )(x)
    x = layers.BatchNormalization(name="bn_conv2")(x)
    x = layers.MaxPooling1D(pool_size=2, name="maxpool_2")(x)

    # BiLSTM block
    x = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=False, kernel_regularizer=l2),
        name="bilstm",
    )(x)
    x = layers.Dropout(dropout_rate, name="drop_lstm")(x)
    branch_a = x

    # ------------------------------------------------------------------
    # Branch B: Statistical features input
    # ------------------------------------------------------------------
    stat_input = keras.Input(shape=(NUM_FEATURES,), dtype="float32", name="stat_input")

    y = layers.Dense(dense_units, activation="relu", kernel_regularizer=l2, name="stat_dense_1")(stat_input)
    y = layers.BatchNormalization(name="bn_stat1")(y)
    y = layers.Dropout(dropout_rate, name="drop_stat1")(y)

    y = layers.Dense(dense_units // 2, activation="relu", kernel_regularizer=l2, name="stat_dense_2")(y)
    y = layers.BatchNormalization(name="bn_stat2")(y)
    y = layers.Dropout(dropout_rate * 0.5, name="drop_stat2")(y)
    branch_b = y

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------
    merged = layers.Concatenate(name="merge")([branch_a, branch_b])

    z = layers.Dense(dense_units, activation="relu", kernel_regularizer=l2, name="merged_dense_1")(merged)
    z = layers.BatchNormalization(name="bn_merge1")(z)
    z = layers.Dropout(dropout_rate, name="drop_merge1")(z)

    z = layers.Dense(dense_units // 2, activation="relu", kernel_regularizer=l2, name="merged_dense_2")(z)
    z = layers.BatchNormalization(name="bn_merge2")(z)
    z = layers.Dropout(dropout_rate * 0.5, name="drop_merge2")(z)

    output = layers.Dense(1, activation="sigmoid", name="output")(z)

    model = keras.Model(inputs=[char_input, stat_input], outputs=output, name="PhishNet")

    # ------------------------------------------------------------------
    # Compile with Adam + gradient clipping
    # ------------------------------------------------------------------
    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=1.0,  # gradient clipping
    )
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )

    return model


def get_training_callbacks(checkpoint_dir: str = "saved_model") -> list:
    """
    Return a list of Keras callbacks implementing optimization best practices.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    return [
        # Early stopping – prevent overfitting
        callbacks.EarlyStopping(
            monitor="val_auc",
            patience=7,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),
        # Learning rate scheduling – reduce on plateau
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        # Model checkpoint – save best weights
        callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, "phishnet_best.keras"),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        # TensorBoard logging (optional)
        callbacks.TensorBoard(
            log_dir=os.path.join(checkpoint_dir, "logs"),
            histogram_freq=1,
        ),
    ]


def model_summary_to_str(model: keras.Model) -> str:
    """Capture model.summary() as a string."""
    lines = []
    model.summary(print_fn=lambda line: lines.append(line))
    return "\n".join(lines)
