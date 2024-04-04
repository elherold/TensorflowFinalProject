from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


import json
import numpy as np
import pickle
import os
import urllib.request
import zipfile

def create_embedding_matrix(word_index, embedding_dim=100):
    """
    Creates an embedding matrix for the Embedding Layer from the GloVe embeddings.
    Downloads the embeddings if they don't exist and extracts the desired file.

    Parameters:
    word_index(dict): A dictionary mapping words to their index in the Tokenizer. 
    embedding_dimension (int): The dimension of the embedding layer.

    Returns: test change
    numpy.ndarray: An embedding matrix where the ith row gives the embedding of the word with index i.
    """

    # Define base directory and filename
    data_dir = '../data'  # Modify this to your desired data directory
    glove_file = 'glove.6B.100d.txt'

    # Construct full embedding file path
    embedding_file = os.path.join(data_dir, glove_file)

    if not os.path.exists(embedding_file):
        print("Downloading GloVe embeddings... hang on this may take a few minutes")
        urllib.request.urlretrieve('https://nlp.stanford.edu/data/glove.6B.zip', 'glove.6B.zip')  # Download zip file

        with zipfile.ZipFile('glove.6B.zip', 'r') as zip_ref:
            # Extract only the desired file
            for info in zip_ref.infolist():
                if info.filename == glove_file:
                    zip_ref.extract(info, data_dir)
                    break  # Exit the loop after extracting the target file

        # Delete the zip file
        os.remove('glove.6B.zip')
    else:
        print("found existing GloVe embeddings")

    # Load the GloVe embeddings
    embeddings_index = {}
    with open(embedding_file, encoding='utf-8') as f:
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

def tokenizer_padding(X_train, X_test, name, max_length=[500]):
    """
    Tokenizes and pads the training and testing text sequences.
    Loads the tokenizer and word_index if they exist, otherwise creates them.
    """
    
    try:
        embedding_matrix = np.load(f'../models/embedding_matrix_{name}.npy')
        word_index = json.load(open(f'../models/word_index_{name}.json'))
        tokenizer = pickle.load(open(f'../models/tokenizer_{name}.pickle', 'rb'))
    except:
        # Initialize the Tokenizer
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(X_train + X_test) # Combine to ensure tokenizer covers both sets of words

        # Prepare the embedding matrix
        word_index = tokenizer.word_index
        embedding_matrix = create_embedding_matrix(word_index)
    
    # Tokenize and pad the sequences
    train_sequence = tokenizer.texts_to_sequences(X_train)
    test_sequence = tokenizer.texts_to_sequences(X_test)
    # Pad the sequences with specified max length
    X_train_padded = [pad_sequences(train_sequence, maxlen=length) for length in max_length]
    X_test_padded = [pad_sequences(test_sequence, maxlen=length) for length in max_length]
    print("Tokenization completed")

    return X_train_padded, X_test_padded, tokenizer, word_index, embedding_matrix