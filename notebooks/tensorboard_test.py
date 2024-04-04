import tensorflow as tf

model_path = '../models/model_augmented_synonyms'

# load the model
model = tf.keras.models.load_model(model_path)

model.summary()