import numpy as np
import json

names = ["original_data", "augmented_synonyms"]
names_wi = ["word_index_original_data", "word_index_augmented_synonyms"]

# Load the embedding matrices and word indices
embedding_matrix_paths = [f"../models/embedding_matrix_{name}.npy" for name in names]
embedding_matrices = [np.load(path) for path in embedding_matrix_paths]
word_index_paths = [f"../models/word_index_{name}.json" for name in names_wi]
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
def compute_aggregate_vector(words, embeddings):
    vectors = [embeddings[word] for word in words if word in embeddings]
    return np.mean(vectors, axis=0) if vectors else None

def compute_neutral_direction_for_axis(seed_word_sets, embedding_matrix):
    aggregate_vectors = [compute_aggregate_vector(words, embedding_matrix) for words in seed_word_sets]
    print(f"Aggregate vectors calculated: {aggregate_vectors}")
    neutral_direction = np.mean(aggregate_vectors, axis=0)
    print(f"Neutral direction calculated: {neutral_direction}")
    return neutral_direction / np.linalg.norm(neutral_direction)

def multi_axis_debiasing(embedding_matrix, seed_sets):
    debiased_embedding_matrix = np.copy(embedding_matrix)  # Make a copy to avoid altering the original embeddings
    for axis, seed_word_sets in seed_sets.items():
        neutral_direction = compute_neutral_direction_for_axis(seed_word_sets, debiased_embedding_matrix)
        for i, embedding in enumerate(debiased_embedding_matrix):
            projection = np.dot(embedding, neutral_direction) * neutral_direction
            debiased_embedding_matrix[i] -= projection
    return debiased_embedding_matrix

def main(embedding_matrices, word_indices, seed_sets):
    for name, embedding_matrix in zip(names, embedding_matrices):
        debiased_embedding_matrix = multi_axis_debiasing(embedding_matrix, seed_sets)
        np.save(f"debiased_embedding_matrix_{name}.npy", debiased_embedding_matrix)
