# TensorflowFinalProject
Mitigating Discriminating Bias in Hate-speech Classification

## Tensorboard
run tensorboard.ipynb to view tensorboard

if not working, run the following command in terminal and visit http://localhost:6006/

tensorboard --logdir logs/augmented_all_combined/epoch_1_10_20240404-190119

## Hyperparameter Tuning
This code utilizes KerasTuner to optimize a Long Short-Term Memory (LSTM) model for sentiment analysis. It constructs the model architecture with embedding layer using for text data, an LSTM layer to capture sequential information, dropout for regularization, and a final classification layer. The key aspect is finding the best hyperparameter settings for this model. Here, the code employs a Hyperband search to identify the optimal number of LSTM units, dropout rate, and learning rate that maximize validation accuracy. Notably, it explores different input sequence lengths by padding the training data, allowing the search to consider the impact of sequence length on performance.

Hyperparameter Tuning for Sentiment Analysis LSTM with KerasTuner

This code utilizes KerasTuner to optimize a sentiment analysis LSTM model. The model employs an embedding layer for text processing, an LSTM layer to capture sequential information, dropout for regularization, and a final classification layer.

The core objective is to find the optimal hyperparameters for optimal performance. Hyperparameters control the model's learning process and significantly impact results. Here, KerasTuner's Hyperband search explores combinations of key hyperparameters:

Number of LSTM units: This value determines the model's ability to learn complex relationships within text sequences. The code searches within a specific range (e.g., 16-128) to find the ideal balance between complexity and efficiency. Too few units might miss crucial details, while too many could lead to overfitting.
Dropout rate: This hyperparameter controls the amount of random neuron dropout during training, preventing overfitting. The search space might explore values like 0.1, 0.2, 0.3, allowing the tuner to identify the most effective dropout level.
Learning rate: This parameter controls how quickly the model updates its weights during training. The code might search through values like 0.01, 0.001 to find the optimal learning rate.
Flexibility through Search Space Customization:

You can easily modify the search space for each hyperparameter. For example, if you want wider searchspace of LSTM units (e.g., 32-256), simply adjust the hp.Int call within the model_builder function. Similarly, you can explore different dropout rate ranges or introduce new hyperparameters for tailored hyperparameter searches specific to your data and model requirements.
