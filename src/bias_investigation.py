import numpy as np
import json

names = ["original_data", "augmented_synonyms"]

# Load the embedding matrices and word indices
embedding_matrix_paths = [f"../models/embedding_matrix_{name}.npy" for name in names]
embedding_matrices = [np.load(path) for path in embedding_matrix_paths]
word_index_paths = [f"../models/word_index_{name}.json" for name in names]
word_indices = [json.load(open(path)) for path in word_index_paths]

# Defining the seed words for the different informative dimensions
# Assuming word_index is a dictionary mapping words to their indices in the embedding matrix
male_words = ['man', 'he']
female_words = ['woman', 'she']
seed_sets = {
    'gender': [['man', "he", "his", "him", "boy"], ["woman", "she", "her", "girl"]],
    'religion': [['christian', 'church', 'bible'], ['muslim', 'mosque', 'quran'], ['jewish', 'synagogue', 'torah']],
    'ethnicity': [['white', 'caucasian'], ['black', 'african']],
    'sexuality': [['gay', 'bisexual', 'lesbian', 'queer'], ['straight']]
}

# tests: 
print(f"Word indices of seed_set words: {[word_indices[0][word] for word in seed_sets['gender'][0]]}")
#print(f"Word embeddings of seed_set words: {[embedding_matrices[0][word_indices[0][word]] for word in seed_sets['gender'][0]]}")

def compute_aggregate_vector(words, embeddings, word_index):
    """
    Calculate the average vector for a list of words.

    Parameters:
    - words (list of str): The words to compute the aggregate vector for.
    - embeddings (numpy.ndarray): The embedding matrix where each row corresponds to a word vector.
    - word_index (dict): A dictionary mapping words to their indices in the embedding matrix.

    Returns:
    - numpy.ndarray: The mean vector of the specified words if they exist in the word index, otherwise None.
    """
    print(f"Words to compute aggregate vector for: {words}")
    vectors = [embeddings[word_index[word]] for word in words]
    print("vectors computed for words")
    return np.mean(vectors, axis=0) if vectors else None

def compute_neutral_direction_for_axis(seed_word_sets, embedding_matrix, word_index):
    """
    Compute a neutral direction vector for a given axis based on seed word sets.

    Parameters:
    - seed_word_sets (list of list of str): A list of lists, where each inner list contains seed words defining a concept axis.
    - embedding_matrix (numpy.ndarray): The embedding matrix for the model.
    - word_index (dict): A dictionary mapping words to their indices in the embedding matrix.

    Returns:
    - numpy.ndarray: A normalized vector representing the neutral direction for the axis.
    """
    aggregate_vectors = [compute_aggregate_vector(words, embedding_matrix, word_index) for words in seed_word_sets]
    print(f"Aggregate vectors calculated.")
    neutral_direction = np.mean(aggregate_vectors, axis=0)
    print(f"Neutral direction calculated.")
    return neutral_direction / np.linalg.norm(neutral_direction)

def multi_axis_debiasing(embedding_matrix, seed_sets, word_index):
    """
    Perform debiasing of the embedding matrix across multiple axes defined by seed word sets.

    Parameters:
    - embedding_matrix (numpy.ndarray): The original embedding matrix to be debiased.
    - seed_sets (dict): A dictionary where keys are axis names (e.g., 'gender') and values are lists of seed word lists.
    - word_index (dict): A dictionary mapping words to their indices in the embedding matrix.

    Returns:
    - numpy.ndarray: The debiased embedding matrix.
    """
    debiased_embedding_matrix = np.copy(embedding_matrix)  # Make a copy to avoid altering the original embeddings
    for axis, seed_word_sets in seed_sets.items():
        print(f"Currently working on debiasing for axis {axis}.")
        neutral_direction = compute_neutral_direction_for_axis(seed_word_sets, debiased_embedding_matrix, word_index)
        for i, embedding in enumerate(debiased_embedding_matrix):
            projection = np.dot(embedding, neutral_direction) * neutral_direction
            debiased_embedding_matrix[i] -= projection
    return debiased_embedding_matrix

def main(embedding_matrices, word_indices, seed_sets):
    for i, (name, embedding_matrix) in enumerate(zip(names, embedding_matrices)):
        debiased_embedding_matrix = multi_axis_debiasing(embedding_matrix, seed_sets, word_indices[i])
        np.save(f"../models/debiased_embedding_matrix_{name}.npy", debiased_embedding_matrix)

if __name__ == "__main__":
    main(embedding_matrices, word_indices, seed_sets)
