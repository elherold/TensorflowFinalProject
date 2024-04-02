from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import TensorBoard
import pandas as pd
import numpy as np
import json
import pickle
from merge_train_test import getting_datasets
import os
from datetime import datetime

def create_embedding_matrix(word_index, embedding_dim=100):
    """
    Creates an embedding matrix for the Embedding Layer from the GloVe embeddings.

    Parameters:
    word_index(dict): A dictionary mapping words to their index in the Tokenizer. 
    embedding_dimension (int): The dimension of the embedding layer.

    Returns:
    numpy.ndarray: An embedding matrix where the ith row gives the embedding of the word with index i.
    """
    # Load the GloVe embeddings
    embeddings_index = {}
    with open('../data/glove.6B.100d.txt', encoding='utf-8') as f:
        # Fill up the embedding matrix dictionary with the natural language words as keys and their respective vector representations as values
        print(type(word_index))
        l = 0
        for line in f:
            l += 1
            if l % 1000 == 0:
                print(f"Processed {l} lines")
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    
    # Prepare embedding matrix
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        if i % 1000 == 0:
            print(f"Processed {i} words")
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in the embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

def tokenizer_padding(X_train, X_test):
    """
    Tokenizes and pads the training and testing text sequences.
    """
    # Initialize counters for numeric values
    numeric_values_count_train = 0

    # Filter out numeric values and count their occurrences
    X_train_filtered = []
    for i, x in enumerate(X_train):
        if isinstance(x, str):
            X_train_filtered.append(x)
        else:
            numeric_values_count_train += 1
            
    print(f"Number of NaN values in training set: {numeric_values_count_train}, it is at index {i}")

    # Initialize the Tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train_filtered + X_test) # Combine to ensure tokenizer covers both sets of words

    print(f"type of X-training: {type(X_train_filtered)}")
    print(f"type of X-testing: {type(X_test)}")
    # Tokenize and pad the sequences
    train_sequence = tokenizer.texts_to_sequences(X_train_filtered)
    test_sequence = tokenizer.texts_to_sequences(X_test)
    max_length = 500 # Maximum sequence length we are using (this means some comments get cut off, as the longest comment is 1250 words)
    X_train_padded = pad_sequences(train_sequence, maxlen=max_length)
    X_test_padded = pad_sequences(test_sequence, maxlen=max_length)

    # Prepare the embedding matrix
    word_index = tokenizer.word_index
    embedding_matrix = create_embedding_matrix(word_index)
    print("Tokenization completed")

    return X_train_padded, X_test_padded, tokenizer, word_index, embedding_matrix

train_sets, X_test, y_test = getting_datasets()

for name, (X_train, y_train) in train_sets.items():
    # Tokenize and pad the training and testing sequences
    X_train = list(X_train)

    # This needs to be removed eventually, just skipping it for testing purposes
    if name=="original_data" or name=="augmented_synonyms":
        continue
        
    X_train_padded, X_test_padded, tokenizer, word_index, embedding_matrix = tokenizer_padding(X_train, X_test)

    y_train = np.array(y_train, dtype=np.int32)
    y_test = np.array(y_test, dtype=np.int32)

    print(f"Currently training model {name}")
    print("X_train_padded:", X_train_padded.shape, X_train_padded.dtype)
    print("y_train:", y_train.shape, y_train.dtype)
    print("X_test_padded:", X_test_padded.shape, X_test_padded.dtype)
    print("y_test:", y_test.shape, y_test.dtype)

    
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
    history = model.fit(X_train_padded, y_train, validation_data=(X_test_padded, y_test), epochs=3, batch_size=64)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test_padded, y_test)
    print(f"Model {name} - Loss: {loss}, Accuracy: {accuracy}")

    # After training, save the tokenizer and word_index
    with open(f'../models/tokenizer_{name}.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'../models/word_index_{name}.json', 'w') as f:
        json.dump(word_index, f)

    # Save the embedding matrix
    np.save(f'../models/embedding_matrix_{name}.npy', embedding_matrix)

    # Save the model
    model.save(f'../models/model_{name}')

