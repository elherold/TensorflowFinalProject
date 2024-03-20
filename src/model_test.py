
# create the model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
import pandas as pd
from merge_train_test import getting_datasets


dict_datasets, max_sequence_len, word_index = getting_datasets()


for key, value in dict_datasets.items():
    print(f"Key: {key}, Value: {value}")

    X_train, X_test, y_train, y_test = value

    model = Sequential()
    model.add(Embedding(input_dim=len(word_index) + 1, output_dim=100, input_length=max_sequence_len))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))  # Dropout zur Reduzierung von Overfitting
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=4, batch_size=32, validation_data=(X_test, y_test))
    history_dict = history.history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(f'../models/{key}_history.csv')
    model.save('../models/{key}_model')
    print(model.summary())


