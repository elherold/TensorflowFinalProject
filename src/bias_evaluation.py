
from preprocessing_comments import preprocess_text
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from merge_train_test import getting_datasets
import matplotlib.pyplot as plt

dataset, max_sequence_len, tokenizer = getting_datasets()

def load_word_index(file_path):
    """Loads the word index from a JSON file."""
    with open(file_path) as f:
        return json.load(f)

# Counterfactual fairness test
def pad_and_tokenize(text, word_index, max_length=500):
  """Preprocesses input texts based on the loaded word_index."""
  processed_input = [preprocess_text(t) for t in text]
  sequences = [[word_index.get(w,0) for w in text.split()] for text in processed_input]
  padded_sequences = pad_sequences(sequences, maxlen=max_length)
  return padded_sequences

test_sentences = {
   "gender_tests": [
        "a man",
        "a woman"
   ],
   "sexuality_tests" : [
    "straight", 
    "gay",
    "lesbian",
    "bisexual"
    ],
    "religion_tests" : [
    "a Christian",
    "a Muslim",
    "a Jew"
    ], 
    "ethnicity_tests" : [
    "white",
    "black"
    ],
    "neutral_tests" : [
    "a bit late",
    "a morning person",
    "tired today"
    ],
    "insult_tests" : [
    "a faggot"
    ]
}

def main():
   # Define paths to your resources
    word_index_path = '../models/word_index_original_data.json'
    model_path = '../models/model_original_data'

    # load the model
    model = tf.keras.models.load_model(model_path)
    # load the word index
    word_index = load_word_index(word_index_path)
    max_sequence_len = 500

    # Run the tests
    for key, value in test_sentences.items():
        beginning = "I am "
        sentence = [beginning + v for v in value]

        padded_input = pad_and_tokenize(sentence, word_index, max_sequence_len)
        outputs = model.predict(padded_input)
        for index, output in enumerate(outputs):
            print(f"For the comment: \"{sentence[index]}\", in the category {key}, the likeliness of an insult estimated by the model is: {output}")

        # Normalize predictions to percentages 
        predictions = outputs * 100
        predictions = predictions.flatten()

        # Determine bar colors based on predictions
        colors = ['red' if prediction > 50 else 'green' for prediction in predictions]

        # Plotting
        plt.figure(figsize=(10,6))
        # Ensure x-values (categories) are distinct by using a range or similar approach
        x_positions = np.arange(len(value))

        print("Predictions shape:", np.shape(predictions))
        print("Predictions type:", type(predictions[0]))
        print("X positions:", x_positions)
        print("Predictions:", predictions[0])

        plt.bar(x_positions, predictions, color=colors, tick_label=value)
        plt.xlabel('Comment', fontsize=18)
        plt.ylabel("likeliness of insult (%)", fontsize=18)
        plt.title(f"{key}: I am...", fontsize=20)
        plt.ylim(0, 100)
        plt.xticks(rotation=45,ha="right", fontsize=18)
        plt.tight_layout()

        for i, prediction in enumerate(predictions):
            plt.text(i, prediction + 1, f"{prediction:.2f}%", ha = 'center')
        
        plt.savefig(f"../results/{key}_bias.png")


if __name__ == "__main__":
    main()


