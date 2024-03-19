import pandas as pd
import random
from random import randrange
from scipy.spatial.distance import cosine
from nltk.corpus import wordnet
import tensorflow_hub as hub
import os

# Load the Universal Sentence Encoder module
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

def find_synonyms(word):
    """
    Find synonyms for a given word using NLTK's WordNet.

    Parameters:
    word (str): The word for which synonyms are to be found.

    Returns:
    list: A list of synonyms.
    """
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)[:10]

def identify_best_synonym(original_embedding, original_sentence, word_to_replace, synonyms):
    """
    Identify the best synonym for a word in a sentence based on cosine similarity.

    Parameters:
    original_embedding (numpy.ndarray): The embedding of the original sentence.
    original_sentence (str): The original sentence.
    word_to_replace (str): The word in the sentence to be replaced.
    synonyms (list): A list of potential synonyms for the word.

    Returns:
    tuple: The best synonym and its cosine distance from the original sentence.
    """
    sentences = [original_sentence.replace(word_to_replace, synonym) for synonym in synonyms]
    embeddings = embed(sentences) # Generate a list of embeddings for a batch of sentences using the Universal Sentence Encoder.

    best_synonym = word_to_replace
    min_distance = float('inf')
    for synonym, new_embedding in zip(synonyms, embeddings):
        distance = cosine(original_embedding, new_embedding)
        if distance < min_distance:
            min_distance = distance
            best_synonym = synonym

    return best_synonym, min_distance

def replace_synonyms(sentence, synonym_cache={}):
    """
    Replace words in a sentence with their synonyms based on minimal cosine distance.

    Parameters:
    sentence (str): The original sentence.
    synonym_cache (dict): Cache to store synonyms to avoid redundant lookups.

    Returns:
    tuple: The modified sentence and the cosine distance of the last replacement.
    """
    if not sentence:
        return sentence, 2  # Return the original sentence if it's empty

    sentence_list = sentence.split()
    num_replacements = round(len(sentence_list) * randrange(10, 25) / 100)
    replacement_indices = random.sample(range(len(sentence_list)), num_replacements)

    original_embedding = embed([sentence])[0]

    for i in replacement_indices:
        word = sentence_list[i]
        if word not in synonym_cache:
            synonym_cache[word] = find_synonyms(word)

        filtered_synonyms = [s for s in synonym_cache[word] if s.lower() != word.lower()]
        if not filtered_synonyms:
            continue

        current_sentence = ' '.join(sentence_list)
        best_synonym, min_distance = identify_best_synonym(original_embedding, current_sentence, word, filtered_synonyms)

        if best_synonym and best_synonym != word:
            sentence_list[i] = best_synonym

    return ' '.join(sentence_list), min_distance

def main():
    """
    Main function to execute the data augmentation process.
    """
    nonzero_targets = pd.read_csv("../data/nonzero_targets.csv").values.tolist()
    augmented_targets = []
    checkpoint = round(len(nonzero_targets)/10)

    for i, (target, sentence) in enumerate(nonzero_targets[:100]):
        augmented_sentence, distance = replace_synonyms(sentence)
        augmented_targets.append((target, augmented_sentence, distance))

        if i % checkpoint == 0:
            print(f"Checkpoint at iteration {i}: current augmented_targets length: {len(augmented_targets)}")

    sorted_list = sorted(augmented_targets, key=lambda x: x[2])
    final_targets_synonyms_first = sorted_list[:5000]
    final_targets_synonyms_middle = sorted_list[5000:10000]

    df_firsts = pd.DataFrame(final_targets_synonyms_first, columns=['Target', "AugmentedSentence", "distance"])
    df_middle = pd.DataFrame(final_targets_synonyms_middle, columns=['Target', "AugmentedSentence", "distance"])
    df_firsts.drop('distance', axis=1, inplace=True)
    df_middle.drop('distance', axis=1, inplace=True)

    filepath_firsts = '../data/synonyms_firsts_test.csv'
    filepath_middle = '../data/synonyms_middle_test.csv'

    if not os.path.exists(filepath_firsts):
        df_firsts.to_csv(filepath_firsts, index=False)
        print("synonyms_firsts.csv saved")
    else:
        print("synonyms_firsts.csv already exists")

    if not os.path.exists(filepath_middle):
        df_middle.to_csv(filepath_middle, index=False)
        print("synonyms_middle.csv saved")  
    else:
        print("synonyms_middle.csv already exists")
     

if __name__ == "__main__":
    main()
