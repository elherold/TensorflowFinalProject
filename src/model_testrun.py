from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import json
from merge_train_test import getting_datasets

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
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    
    # Prepare embedding matrix
            embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
            for word, i in word_index.items():
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    # Words not found in the embedding index will be all-zeros.
                    embedding_matrix[i] = embedding_vector

    return embedding_matrix

def tokenizer_padding(X_train, X_test):
    """
    Tokenizes and pads the training and testing text sequences.
    """

    # Initialize the Tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train + X_test) # Combine to ensure tokenizer covers both sets of words

    # Tokenize and pad the sequences
    train_sequence = tokenizer.texts_to_sequences(X_train)
    test_sequence = tokenizer.texts_to_sequences(X_test)
    max_length = 500 # Maximum sequence length we are using (this means some comments get cut off, as the longest comment is 1250 words)
    X_train_padded = pad_sequences(train_sequence, maxlen=max_length)
    X_test_padded = pad_sequences(test_sequence, maxlen=max_length)

    # Prepare the embedding matrix
    word_index = tokenizer.word_index
    embedding_matrix = create_embedding_matrix(word_index)
    print("Tokenization completed")

    return X_train_padded, X_test_padded, word_index, embedding_matrix

train_sets, X_test, y_test = getting_datasets()

for name, (X_train, y_train) in train_sets.items():
    # Tokenize and pad the training and testing sequences
    X_train_padded, X_test_padded, word_index, embedding_matrix = tokenizer_padding(X_train, X_test)
    print(f"Currently training model {name}")
    
    # Create the model
    model = Sequential()
    model.add(Embedding(input_dim=len(word_index) + 1, output_dim=100, input_length=500, weights=[embedding_matrix], trainable=False)) # set trainable to False to keep the embeddings fixed
    model.add(LSTM(100))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train_padded, y_train, validation_data=(X_test_padded, y_test), epochs=5, batch_size=64)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test_padded, y_test)
    print(f"Model {name} - Loss: {loss}, Accuracy: {accuracy}")

    # Save the model
    model.save(f'../models/model_{name}')

    # Save the word index
    with open(f'../models/word_index_{name}.json', 'w') as f:
        json.dump(word_index, f)

    # Save the embedding matrix
    np.save(f'../models/embedding_matrix_{name}.npy', embedding_matrix)

