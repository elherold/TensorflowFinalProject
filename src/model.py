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
from preprocess_embedding_padding import create_embedding_matrix, tokenizer_padding

def train_models():
    """
    function to train the models.
    """
    # Get the training and testing sets
    train_sets, X_test, y_test = getting_datasets()

    # train a distinct model with each training set
    for name, (X_train, y_train) in train_sets.items():
        print(f"currently working on model: {name}")
        # Tokenize and pad the training and testing sequences
        X_train = list(X_train)

        # This needs to be removed eventually, just skipping it for testing purposes
        if name=="augmented_synonyms" or name=="augmented_backtranslation" or name=="original_data":
            continue
            
        X_train_padded, X_test_padded, tokenizer, word_index, embedding_matrix = tokenizer_padding(X_train, X_test, name)

        y_train = np.array(y_train, dtype=np.int32)
        y_test = np.array(y_test, dtype=np.int32)

        #print(f"Currently training model {name}")
        #print("X_train_padded:", X_train_padded[0].shape, X_train_padded[0].dtype)
        #print("y_train:", y_train.shape, y_train.dtype)
        #print("X_test_padded:", X_test_padded[0].shape, X_test_padded[0].dtype)
        #print("y_test:", y_test.shape, y_test.dtype)

        # Load the model if it already exists
        try: 
            model = tf.keras.models.load_model(f'../models/model_{name}')
            print(f"Model {name} already exists, skipping training")

        except OSError:
            print(f"Model {name} does not exist, training and saving new model")

            # Create the model
            model = Sequential()
            model.add(Embedding(input_dim=len(word_index) + 1, output_dim=100, input_length=500, weights=[embedding_matrix], trainable=False)) # set trainable to False to keep the embeddings fixed
            model.add(LSTM(100))
            model.add(Dropout(0.2))
            model.add(Dense(1, activation='sigmoid'))

            # Compile the model
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            # TensorBoard setup
            log_dir = os.path.join("logs", name, datetime.now().strftime("%Y%m%d-%H%M%S"))
            tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

            # Train the model
            history = model.fit(X_train_padded, y_train, 
                                validation_data=(X_test_padded, y_test), 
                                epochs=3, batch_size=64, 
                                callbacks=[tensorboard_callback])

            # After training, save the tokenizer and word_index
            with open(f'../models/tokenizer_{name}.pickle', 'wb') as handle:
                pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open(f'../models/word_index_{name}.json', 'w') as f:
                json.dump(word_index, f)

            # save the training history
            with open(f'../models/history_{name}.json', 'w') as f:
                json.dump(history.history, f)

            # Save the embedding matrix
            np.save(f'../models/embedding_matrix_{name}.npy', embedding_matrix)

            # Save the model
            model.save(f'../models/model_{name}')
        
        # Evaluate the model
        loss, accuracy = model.evaluate(X_test_padded, y_test)
        print(f"Model {name} - Loss: {loss}, Accuracy: {accuracy}")

if __name__ == "__main__":
    train_models()
