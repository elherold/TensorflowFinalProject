
from tensorflow.keras.preprocessing.text import Tokenizer
from tsensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import random


def is_nan(x):
    return isinstance(x, float) 


def create_balanced_sets(name, input_data, original_zeros):
    """
    Create a balanced dataset by combining input data with a random selection of 'zero' labeled data.

    Parameters:
    name (str): Name of the dataset.
    input_data (list): The list of data with 'one' labels.
    original_zeros (list): The list of data with 'zero' labels.

    Returns:
    tuple: Two lists, one with balanced sentences and another with corresponding labels.
    """
    length = len(input_data)
    included_zeros = random.sample(original_zeros, length)  # Change to sample to avoid duplicates
    balanced_data = included_zeros + input_data
    random.shuffle(balanced_data)
    #print(f"First few balanced_datapoints: {balanced_data[:5]}")
    balanced_sentences = [sentence for _, sentence in balanced_data]
    balanced_targets = [int(label) for label, _ in balanced_data]
    #print(f"In the balanced dataset {name}, there are {len(original_zeros)} zero and {len(input_data)} one labeled datapoints.")
    #print(f"In the balanced dataset {name}, there are {len(balanced_sentences)}  datapoints.")
    #print(f"Debug - {name}:")
    #print(f"First few balanced_sentences: {balanced_sentences[:5]}")
    #print(f"First few balanced_targets: {balanced_targets[:5]}")
    return list(balanced_sentences), list(balanced_targets)

def add_augmented_data(input, name, original_zeros, X_train, y_train):
    """
    Augment the training data with additional balanced data. Test

    Parameters:
    input (list): New data to be added.
    name (str): Name of the dataset being augmented.
    original_zeros (list): List of data with 'zero' labels for balancing.
    X_train (list): Original training sentences.
    y_train (list): Original training labels.

    Returns:
    tuple: Two lists, one with augmented training sentences and another with corresponding labels.
    """
    balanced_sentences, balanced_targets = create_balanced_sets(name, input, original_zeros)
    X_train_aug = X_train + balanced_sentences
    y_train_aug = y_train + balanced_targets

    combined = list(zip(X_train_aug, y_train_aug))
    random.shuffle(combined)
    X_train_aug, y_train_aug = zip(*combined)

    return X_train_aug, y_train_aug

def create_train_test_sets(inputs, original_nonzeros, original_zeros, test_size=0.2, random_state=42):
    """
    Create training and testing sets, ensuring balanced and shuffled datasets.
    It creates multiple different training sets, one with the original comments and then additional ones where augmented training data was added for enhanced performance.
    The augmented data was NOT added to the testing set of course.

    Parameters:
    inputs (dict): Dictionary of input data for augmentation.
    original_nonzeros (list): List of data with 'one' labels.
    original_zeros (list): List of data with 'zero' labels.
    test_size (float): Proportion of dataset to include in the test split.
    random_state (int): Random state for reproducibility.

    Returns:
    tuple: Training sets, X_test, y_test.
    """
    train_sets = {}

    # Create initial balanced dataset
    balanced_sentences, balanced_targets = create_balanced_sets("original_data", original_nonzeros, original_zeros)
    #print(f"Debug - After create_balanced_sets in create_train_test_sets:")
    #print(f"First few X_train: {balanced_sentences[:5]}")
    #print(f"First few y_train: {balanced_targets[:5]}")
    X_train, X_test, y_train, y_test = train_test_split(
        balanced_sentences, balanced_targets, test_size=test_size, random_state=random_state
    )
    #print(f"Initial size of X_train {len(X_train)} and y_train {len(y_train)}")
    train_sets["original_data"] = [X_train, y_train]
    print(f"In the original dataset, there are {len(X_train)} comment datapoints and {len(y_train)} target datapoints in total.")

    # Augment training data with additional datasets
    for name, input in inputs.items():
        X_train_aug, y_train_aug = add_augmented_data(input, name, original_zeros, X_train, y_train)
        print(f"In the augmented dataset {name}, there are {len(X_train_aug)} comment datapoints and {len(y_train_aug)} target datapoints in total.")
        train_sets[name] = [X_train_aug, y_train_aug]

    return train_sets, X_test, y_test

def tokenizer_padding(X_train, y_train, X_test, y_test):
    """
    Tokenizes and pads the training and testing text sequences, and converts labels to numpy arrays.

    This function takes the training and testing text data (X_train, X_test), tokenizes these texts into sequences 
    of integers, and then pads these sequences to ensure that they have the same length. Additionally, it converts 
    the label lists (y_train, y_test) into numpy arrays for use in machine learning models.

    Parameters:
    X_train (list of str): The list of training sentences.
    y_train (list): The list of training labels.
    X_test (list of str): The list of testing sentences.
    y_test (list): The list of testing labels.

    Returns:
    tuple: Returns four elements:
        - X_train_padded (numpy.ndarray): Padded sequences for the training set.
        - y_train_array (numpy.ndarray): Array of training labels.
        - X_test_padded (numpy.ndarray): Padded sequences for the testing set.
        - y_test_array (numpy.ndarray): Array of testing labels.

    The padding is done based on the maximum sequence length found in either the training or testing sets.
    """
    # tokenize sentences
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train) # fit only training data

    train_sequence = tokenizer.texts_to_sequences(X_train)
    test_sequence = tokenizer.texts_to_sequences(X_test)

    # Pad the sequences
    max_sequence_length = max(max(len(x) for x in train_sequence), max(len(x) for x in test_sequence))
    X_train_padded = pad_sequences(train_sequence, maxlen=max_sequence_length)
    X_test_padded = pad_sequences(test_sequence, maxlen=max_sequence_length)

    # Convert y_train to a numpy array
    y_train_array = np.array(y_train)

    # Same for y_test if it's not already a numpy array
    y_test_array = np.array(y_test)
    
    return X_train_padded, y_train_array, X_test_padded, y_test_array


def getting_datasets():

    # Define input DataFrames
    input_dfs = {
        "nonzero_synonyms_first": list(zip(pd.read_csv('../data/synonyms_firsts.csv')['Target'], pd.read_csv('../data/synonyms_firsts.csv')['AugmentedSentence'])),
        "nonzero_synonyms_middle": list(zip(pd.read_csv('../data/synonyms_middle.csv')['Target'], pd.read_csv('../data/synonyms_middle.csv')['AugmentedSentence']))
    }

    # Read original nonzeros
    original_nonzeros = list(zip(pd.read_csv('../data/nonzero_targets.csv')['Target'], pd.read_csv('../data/nonzero_targets.csv')['AugmentedSentence']))

    # Read original zeros
    original_zeros = list(zip(pd.read_csv('../data/zero_targets.csv')['Target'], pd.read_csv('../data/zero_targets.csv')['Comment']))

    # getting rid of any NaN values
    original_zeros = [x for x in original_zeros if not is_nan(x[1])]

    # Create training and testing sets
    train_sets, X_test, y_test = create_train_test_sets(input_dfs, original_nonzeros, original_zeros)

    # Dictionary to store the datasets
    datasets = {}

    # Process each training set
    for name, (X_train, y_train) in train_sets.items():
        # Tokenize and pad the training and testing sequences
        X_train_padded, y_train_array, X_test_padded, y_test_array = tokenizer_padding(X_train, y_train, X_test, y_test)
        
        # Add the datasets to the dictionary
        datasets[name] = [X_train_padded, y_train_array, X_test_padded, y_test_array]

    print("Data processing completed.")
    return datasets