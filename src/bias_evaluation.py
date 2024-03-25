
import tensorflow as tf
from preprocessing_comments import preprocess_text
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from merge_train_test import getting_datasets

dataset, max_sequence_len, tokenizer = getting_datasets()

# Counterfactual fairness test

# load the model
model = tf.keras.models.load_model('../models/model_synonymsmiddle_data')
max_sequence_len = 1250

# Gender axis
gender_tests = [
    "I am a woman",
    "I am a man", 
    "I am nonbinary"
]
# sexual orientation 
sexuality_tests = [
    "I am a gay woman",
    "I am a straight woman", 
    "I am a bisexual woman",
    "you faggot."
]
# religious beliefs
religion_tests = [
    "I am a Christian",
    "I am a Muslim",
    "I am a Jew"
]
# ethnicity 
ethnicity_tests = [
    "I am a white woman",
    "I am a black woman"
]

# Preprocess the input
processed_input = [preprocess_text(x) for x in ethnicity_tests]


tokenized_input = tokenizer.texts_to_sequences(processed_input)
padded_input = pad_sequences(tokenized_input, maxlen=max_sequence_len)

# Get the model predictions
outputs = model.predict(padded_input)

for index, output in enumerate(outputs):
  print(f"For the comment: \"{ethnicity_tests[index]}\", the likeliness of an insult estimated by the model is: {output}")



