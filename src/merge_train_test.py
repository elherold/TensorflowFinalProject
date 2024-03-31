

from sklearn.model_selection import train_test_split
import pandas as pd
import random

print("test if its working, check!")

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


def getting_datasets():

    # Define input DataFrames
    input_dfs = {
        "augmented_synonyms": list(zip(pd.read_csv('../data/synonyms.csv')['Target'], pd.read_csv('../data/synonyms.csv')['AugmentedSentence'])),
        "augmented_backtranslation": list(zip(pd.read_csv('../data/backtranslation.csv')['target'], pd.read_csv('../data/backtranslation.csv')['comment_text'])),
    }

    # Read original nonzeros
    original_nonzeros = list(zip(pd.read_csv('../data/nonzero_targets.csv')['target'], pd.read_csv('../data/nonzero_targets.csv')['comment_text']))

    # Read original zeros
    original_zeros = list(zip(pd.read_csv('../data/zero_targets.csv')['target'], pd.read_csv('../data/zero_targets.csv')['comment_text']))

    # getting rid of any NaN values
    original_zeros = [x for x in original_zeros if not is_nan(x[1])]

    # Create training and testing sets
    train_sets, X_test, y_test = create_train_test_sets(input_dfs, original_nonzeros, original_zeros)

    print("Data processing completed.")
    return train_sets, X_test, y_test


