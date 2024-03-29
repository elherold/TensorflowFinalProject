import pandas as pd
import random
from random import randrange
from scipy.spatial.distance import cosine
from nltk.corpus import wordnet
import nltk
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
    return list(synonyms)[:15]

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
    
    if best_synonym == word_to_replace:
        print(f"Warning: No synonym found for {word_to_replace} in sentence: {original_sentence}")
    print(f"the best synonym for {word_to_replace} is {best_synonym} with a distance of {min_distance}")

    return best_synonym, min_distance

nltk.download('averaged_perceptron_tagger')  # New: Download the NLTK POS tagger
nltk.download('wordnet')

def replace_synonyms(sentence, synonym_cache={}):
    if not sentence:
        return sentence, 2

    # New: Use NLTK's POS tagger to tag each word in the sentence
    sentence_list = nltk.word_tokenize(sentence)
    pos_tags = nltk.pos_tag(sentence_list)

    # Modification for efficiency: Pre-compute number of replacements based on content words
    content_indices = [i for i, (_, pos_tag) in enumerate(pos_tags) if pos_tag.startswith(('N', 'V', 'J', 'R'))]
    num_replacements = round(len(content_indices) * randrange(35, 45) / 100)
    if num_replacements == 0:  # Ensure there's at least one replacement if possible
        num_replacements = min(1, len(content_indices))
    replacement_indices = random.sample(content_indices, num_replacements)

    original_embedding = embed([sentence])[0]

    # Set a default value for min_distance at the start
    min_distance = float('inf')  # Default value

    for i in replacement_indices:
        word, pos_tag = sentence_list[i], pos_tags[i][1]
        if word not in synonym_cache:
            synonym_cache[word] = find_synonyms(word)

        # Filter synonyms for a different base form and check if it's a content word
        filtered_synonyms = [s for s in synonym_cache[word] if s.lower() != word.lower() and nltk.pos_tag([s])[0][1] == pos_tag]
        if not filtered_synonyms:
            continue

        current_sentence = ' '.join(sentence_list)
        best_synonym, min_distance = identify_best_synonym(original_embedding, current_sentence, word, filtered_synonyms)

        # Update the sentence list with the best synonym found
        if best_synonym and best_synonym != word:
            sentence_list[i] = best_synonym
        

        if min_distance == float('inf'):
            print(f"Warning: No synonym found for {word} at index {i} in sentence: {sentence}")

    augmented_sentence = ' '.join(sentence_list)
    if augmented_sentence == sentence:
        print(f"Warning: The augmented sentence is exactly the same as the original: \"{sentence}\"")
        augmented_sentence = None


    # Rejoin the sentence after replacements
    return augmented_sentence, min_distance

def main():
    """
    Main function to execute the data augmentation process.
    """
    nonzero_targets = pd.read_csv("../data/nonzero_targets.csv").values.tolist()
    augmented_targets = []
    checkpoint = round(len(nonzero_targets[:100])/10)

    for i, (target, sentence) in enumerate(nonzero_targets):
        augmented_sentence, distance = replace_synonyms(sentence)
        if augmented_sentence:
            augmented_sentence = augmented_sentence.replace("_", " ")
            augmented_targets.append((target, augmented_sentence, distance))

        if i % checkpoint == 0:
            print(f"Checkpoint at iteration {i}: current augmented_targets length: {len(augmented_targets)}")

    sorted_list = sorted(augmented_targets, key=lambda x: x[2])

    df_firsts = pd.DataFrame(sorted_list, columns=['Target', "AugmentedSentence", "distance"])
    df_firsts.drop('distance', axis=1, inplace=True)

    filepath_firsts = '../data/synonyms_test.csv'

    if not os.path.exists(filepath_firsts):
        df_firsts.to_csv(filepath_firsts, index=False)
        print("synonyms.csv saved")
    else:
        print("synonyms.csv already exists")

if __name__ == "__main__":
    main()
