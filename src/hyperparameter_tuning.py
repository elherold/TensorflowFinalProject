from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
import tensorflow as tf
import keras_tuner as kt

import numpy as np
import os

from merge_train_test import getting_datasets
from src.embedding_padding import create_embedding_matrix, tokenizer_padding

word_index = None  # Initialize as None
embedding_matrix = None  # Initialize as None
INPUT_LENGTH = None


def model_builder(hp):
    """
    Function to build the model for hyperparameter tuning.
    Defines the model architecture and hyperparameters to tune.
    Gets called by the Keras Tuner.
    Args:
        hp (HyperParameters): The hyperparameters object.
    Returns:
        Sequential: The model to be tuned."""
    # Create the model
    model = Sequential()

    global word_index  # Use the global keyword to access the variable
    global embedding_matrix  # Use the global keyword to access the variable
    global INPUT_LENGTH  # Use the global keyword to access the variable

    model.add(
        Embedding(
            input_dim=len(word_index) + 1,
            output_dim=100,
            input_length=INPUT_LENGTH,
            weights=[embedding_matrix],
            trainable=False,
        )
    )  # set trainable to False to keep the embeddings fixed
    # Tune the number of units in the first Dense layer
    # Choose an optimal value between 16-128
    hp_lstm_units = hp.Int("lstm_units", min_value=16, max_value=128, step=32)
    model.add(LSTM(hp_lstm_units))
    # Tune the dropout rate in the Dropout layer
    # Choose an optimal value from 0.1, 0.2, 0.3, 0.4, or 0.5
    hp_dropout = hp.Float("dropout", min_value=0.1, max_value=0.5, step=0.1)
    model.add(Dropout(hp_dropout))
    model.add(Dense(1, activation="sigmoid"))

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def find_best_hyperparameters(
    DataSize=150000, InputLength=200, DataSubset="augmented_all_combined"
):
    """
    Find the best hyperparameters for the model using Keras Tuner.
    Args:
        DataSize (int): The size of the training data.
        InputLength (int): The length of the input sequences.
        DataSubset (str): The name of the data subset to use.
    """
    # Get the training and testing sets
    train_sets, X_test, y_test = getting_datasets()
    DATA_SIZE = DataSize
    global INPUT_LENGTH
    INPUT_LENGTH = InputLength
    DATA_NAME = DataSubset

    # Limit the size of the training data
    if DATA_SIZE > len(train_sets[DATA_NAME][0]):
        DATA_SIZE = len(train_sets[DATA_NAME][0])
        print(f"Data size of {DataSize} is too large, setting it to {DATA_SIZE}")

    X_train = train_sets[DATA_NAME][0][0:DATA_SIZE]
    y_train = train_sets[DATA_NAME][1][0:DATA_SIZE]

    # Tokenize and pad the training and testing sequences
    X_train = list(X_train)

    global word_index  # Use the global keyword to modify the variable

    global embedding_matrix  # Use the global keyword to modify the variable

    X_train_padded, X_test_padded, tokenizer, word_index, embedding_matrix = (
        tokenizer_padding(X_train, X_test, "augmented_all_combined", [100, 200, 500])
    )

    print(X_train_padded[0].shape)
    print(X_train[0])
    print(len(X_train[0]))
    print(X_train_padded[0])
    y_train = np.array(y_train, dtype=np.int32)
    y_test = np.array(y_test, dtype=np.int32)

    # Before tuning, create multiple versions of your input arrays with lengths 100, 200, and 500. You can achieve this using padding or truncation techniques.
    input_arrays = {
        100: X_train_padded[0],
        200: X_train_padded[1],
        500: X_train_padded[2],
    }
    tuner = kt.Hyperband(
        model_builder,
        objective="val_accuracy",
        max_epochs=10,
        factor=3,
        directory=f"../models/tuner/hp_tuner_datalen_{DATA_SIZE}_inputlen_{INPUT_LENGTH}",
        project_name="combined_data",
    )

    stop_early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)

    tuner.search(
        input_arrays[INPUT_LENGTH],
        y_train,
        epochs=50,
        validation_split=0.2,
        callbacks=[stop_early],
    )

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(
        f"""
    The hyperparameter search is complete. The optimal number of LSTM units is {best_hps.get('lstm_units')},
    the optimal dropout rate is {best_hps.get('dropout')},
    and the optimal learning rate for the optimizer is {best_hps.get('learning_rate')}.
    """
    )

    tuner.results_summary()


if __name__ == "__main__":
    find_best_hyperparameters()
