import tensorflow as tf


model = tf.keras.models.load_model('../models/model_synonymsfirst_data')

# Extract the trained embeddings
embeddings = model.layers[0].get_weights()[0]

# Apply Orthogonal Subspace correction
# STEP 0: Define the Concept SUbspaces: Identify vectors that represent your concepts of interest. e.g gender (v1) and occupation (v2)
# STEP 1: Orthogonalization Process: like Gram-Schmidt orthogonolization to adjust the occupation subspace (vs) so that it becomes orthogonal to the gender subspace (v1).
# This involves modifying v2 so that the dot product between v1 and the new v'2 is zero
# STEP 2: Rotate all other datapoints (word embeddings ) towards this new orthogonal setup. Points close to v1 rotate less compared to points close to v2, which rotate to align with the new v2'
# STEP 3: Preserve inherent associations - ensure that inherently related concepts like grandpa/mal retain their meaningful associations through this transformation process. 
import numpy as np

def define_subspaces(embeddings):
    # Example: defining subspaces as mean of selected word embeddings
    gender_words = ['he', 'his', 'him', 'she', 'her', 'hers', 'man', 'woman']
    occupation_words = ['engineer', 'scientist', 'lawyer', 'banker', 'nurse', 'homemaker', 'maid', 'receptionist']
    
    v1 = np.mean([embeddings[word] for word in gender_words], axis=0)
    v2 = np.mean([embeddings[word] for word in occupation_words], axis=0)
    return v1, v2

def make_orthogonal(v1, v2):
    # Adjust v2 to be orthogonal to v1
    v2_prime = v2 - np.dot(v2, v1) / np.dot(v1, v1) * v1
    return v2_prime

def rotate_embeddings(embeddings, v1, v2_prime):
    # Rotate embeddings to align with the new, orthogonal subspace
    # This step is more complex and requires determining the rotation matrix
    # that aligns v2 with v2_prime, and then applying this rotation to all embeddings.
    pass

def oscar_algorithm(embeddings):
    v1, v2 = define_subspaces(embeddings)
    v2_prime = make_orthogonal(v1, v2)
    rotated_embeddings = rotate_embeddings(embeddings, v1, v2_prime)
    return rotated_embeddings
