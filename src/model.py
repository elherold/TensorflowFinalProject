from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf

import numpy as np
import json
import pickle
import os
from datetime import datetime

from merge_train_test import getting_datasets
from embedding_padding import create_embedding_matrix, tokenizer_padding


def train_models():
    """
    function to train the models.
    1. Get the training and testing sets
    2. Train a distinct model with each training set
    3. Load Files
    4. Create the model
    5. Train
    6. Write Files
    7. Evaluate the model
    Adjust Hyperparameters if you want, these were declared best by hyperparametertuning using hyperband
    """
    # Tuned Hyperparameter
    LR = 0.001
    LSTM_UNITS = 112
    DROPOUT = 0.3

    # Manual Hyperparameter
    FINAL = True  # Train the final model on the combined data
    EPOCHS = 0  # How many EPOCHS to train
    START_EPOCH = 30  # Loading exisiting model of specific epoch
    INPUT_LENGTH = 500  # Number of input tokens

    # Get the training and testing sets
    train_sets, X_test, y_test = getting_datasets()

    # train a distinct model with each training set
    for name, (X_train, y_train) in train_sets.items():
        print(f"currently working on model: {name}")
        # Tokenize and pad the training and testing sequences
        X_train = list(X_train)

        # This needs to be removed eventually, just skipping it for testing purposes
        if (
            name == "augmented_synonyms"
            or name == "augmented_backtranslation"
            or name == "original_data"
        ):
            continue

        if FINAL == True:
            name = "augmented_all_combined"

        X_train_padded, X_test_padded, tokenizer, word_index, embedding_matrix = (
            tokenizer_padding(X_train, X_test, name, max_length=[INPUT_LENGTH])
        )

        y_train = np.array(y_train, dtype=np.int32)
        y_test = np.array(y_test, dtype=np.int32)

        ############################################
        #           Load Files
        #############################################

        history_file = f"../models/history_{name}.json"
        existing_history = {}

        # Load the model if it already exists
        try:
            # To get files before and after epoch was added to the name
            try:
                model = tf.keras.models.load_model(f"../models/model_{name}")
            except OSError as e:
                if START_EPOCH > 0:
                    try:
                        model = tf.keras.models.load_model(
                            f"../models/model_{name}_epoch_{START_EPOCH}"
                        )
                    except OSError as e:
                        raise
                else:
                    raise
            if EPOCHS == 0:
                print(
                    f"model_{name}_epoch_{START_EPOCH}' exists, just evaluating model, skipping training"
                )
            else:
                print(
                    f"model_{name}_epoch_{START_EPOCH}' exists, continue training model"
                )

            try:
                with open(history_file, "r") as f:
                    existing_history = json.load(f)
            except FileNotFoundError:
                try:
                    with open(
                        os.path.join(
                            history_file[:-5], f"_epoch_{EPOCHS+START_EPOCH}.json"
                        ),
                        "r",
                    ) as f:
                        existing_history = json.load(f)
                except FileNotFoundError:
                    pass  # If the file doesn't exist, we'll create it later
        except OSError:
            print(f"Model {name} does not exist, training and saving new model")
            START_EPOCH = 0

            ####################
            # Create the model
            ####################
            model = Sequential()
            model.add(
                Embedding(
                    input_dim=len(word_index) + 1,
                    output_dim=100,
                    input_length=INPUT_LENGTH,
                    weights=[embedding_matrix],
                    trainable=False,
                )
            )  # set trainable to False to keep the embeddings fixed
            model.add(LSTM(LSTM_UNITS))
            model.add(Dropout(DROPOUT))
            model.add(Dense(1, activation="sigmoid"))

            # Compile the model
            model.compile(
                loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
                metrics=["accuracy"],
            )

        if EPOCHS > 0:
            ########################
            #       Train
            ########################

            # TensorBoard setup
            time_id = datetime.now().strftime("%Y%m%d-%H%M%S")
            log_dir = os.path.join(
                "logs", name, f"epoch_{START_EPOCH+1}_{EPOCHS+START_EPOCH}_{time_id}"
            )
            tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

            # Train the model
            history = model.fit(
                X_train_padded,
                y_train,
                validation_data=(X_test_padded, y_test),
                epochs=EPOCHS,
                batch_size=64,
                callbacks=[tensorboard_callback],
            )
            ########################
            #       Write Files
            ########################

            if START_EPOCH == 0:
                # After training, save the tokenizer and word_index
                with open(f"../models/tokenizer_{name}.pickle", "wb") as handle:
                    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

                with open(f"../models/word_index_{name}.json", "w") as f:
                    json.dump(word_index, f)

            ########################
            #       Write History
            ############:D##########

            # Get the new history from the current training run
            new_history = history.history

            # Update existing history with new data for each key
            for key in existing_history.keys():
                existing_history[key].extend(new_history[key])

            # save the training history
            with open(history_file, "w") as f:
                json.dump(existing_history, f)

            # Save the embedding matrix
            np.save(f"../models/embedding_matrix_{name}.npy", embedding_matrix)

            # Save the model
            model.save(f"../models/model_{name}_epoch_{EPOCHS+START_EPOCH}")

        ########################
        # Evaluate the model
        ########################

        loss, accuracy = model.evaluate(X_test_padded, y_test)
        print(f"Model {name} - Loss: {loss}, Accuracy: {accuracy}")


if __name__ == "__main__":
    train_models()
