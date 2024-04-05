# TensorflowFinalProject
Leveraging Alterations in Embedding Subspaces to Mitigate Bias in Detecting Harmful Speech using Long Short-Term Memory Models.


## File contents:
----------------------------------------------
### preprocessing_comments.py
This script serves as a preprocessing tool for text data, designed to clean and prepare datasets for our natural language processing task. It efficiently processes texts by converting them to lowercase, removing HTML tags, punctuation, numbers, isolated characters, and excessive spaces. The main function orchestrates the loading of datasets, applies the preprocessing steps to each text entry, and segregates the processed texts into two categories based on their target labels—zero and nonzero targets. These preprocessed datasets are then saved into separate CSV files for further use. This streamlined approach to data preparation ensures the text is in a suitable format for our LSTM model, enhancing their ability to learn from cleaner, more uniform data.

----------------------------------------------
### augmented_synonyms.py
This script implements an advanced data augmentation process utilizing the Universal Sentence Encoder for semantic embedding and NLTK's WordNet for synonym discovery. It is designed to enhance text data by identifying and replacing words with their synonyms in a context-aware manner, minimizing the cosine distance between the original and augmented sentences' embeddings. This approach ensures the generated text retains its original meaning while introducing variability. The script processes sentences from a dataset, selectively replacing content words with their synonyms based on part-of-speech tagging and semantic similarity, aiming to create augmented data that supports the development of more robust natural language processing models. The results are saved into a new CSV file, facilitating easy integration into further workflows. This process is particularly useful for increasing dataset diversity without compromising the semantic integrity of the text, thereby improving model generalization.

----------------------------------------------
### Hyperparameter Tuning
run hyperparameter_tuning.ipynb to tune hyperparameters

----------------------------------------------
### merge_train_test.py
This script is aimed to merge our previously defined datasets as a final preparation step. For performance comparison purposes we train our model separately once on just the original data, on the original data combined with the augmented synonyms, on the original data combined with the augmented backtranslation and on the original data combined with both datasets. Its objective is to create balanced training and testing sets. This means with an equal amount of one labeled training data and 0 labeled training data, mitigating an overfitting towards a tendency of the model to simply always predict one of both classes. It only combines the training data with the augmented datapoints, to see how the augmentation improves performance on the original testing set which remains unchanged. 
By leveraging Python's pandas for data manipulation and sklearn for splitting datasets, the script provides an efficient workflow for preparing data for our subsequent model training as it outputs all train-test splits in a dictionary where the key specifies the current datapoint-combination at hand. 

----------------------------------------------
### embedding_padding.py
This script facilitates the preparation of text data for deep learning models by generating an embedding matrix from GloVe embeddings and tokenizing and padding text sequences. It checks for existing GloVe embeddings and downloads them if necessary, creating an embedding matrix that maps each word in the tokenizer's word index to its embedding vector. This is done in try and except blocks in order to only re-perform the padding and embedding process if necessary. The script supports processing both training and testing datasets, ensuring sequences are appropriately padded for consistent input dimensions. If tokenizers and word indices already exist, the script loads them to maintain consistency across model training sessions; otherwise, it creates new ones based on the provided text data. This structured approach to data preparation enhances model performance by leveraging pre-trained embeddings for a rich representation of text data.

----------------------------------------------
### model.py
This script orchestrates the training of deep learning models for text classification tasks, utilizing the TensorFlow and Keras libraries. It features a process for downloading and integrating GloVe embeddings into an embedding matrix, preparing and padding tokenized text data for training, and dynamically adjusting model hyperparameters based on previously determined optimal settings. The script supports training multiple models on various datasets, including those enhanced with augmented data, and incorporates the capability to resume training from a specific epoch or to initiate training of new models from scratch. It employs a sequential model architecture with layers tailored for text processing, including an Embedding layer loaded with pre-trained GloVe vectors, LSTM units for sequence learning, and Dropout for regularization. Model performance is monitored using TensorBoard, and the training process generates and updates historical data, model checkpoints, and evaluation metrics. This comprehensive training framework is designed to facilitate experimentation with different model configurations and datasets, streamlining the development of robust models for analyzing and classifying textual data.

----------------------------------------------
### Tensorboard
run tensorboard.ipynb to view tensorboard
if not working, run the following command in terminal and visit http://localhost:6006/
tensorboard --logdir logs/augmented_all_combined/epoch_1_10_20240404-190119

----------------------------------------------
### bias_evaluation.py: 
This script assesses the model's fairness by performing counterfactual bias testing, focusing on detecting biases in predictions across demographics like gender, sexuality, religion, and ethnicity. It processes and tokenizes test sentences using a pre-trained tokenizer to maintain consistency with the tokinzation of the trained model. After that, it is evaluating the model's tendency to label these sentences as insults. Results are visualized through color-coded bar charts, indicating the likelihood of each sentence being perceived as offensive. This streamlined analysis helps identify and address potential biases, ensuring the model treats all groups equitably.

----------------------------------------------
### bias_mitigation.py: 
This script is designed to address and mitigate biases within word embeddings, focussing on the same attributes as the bias_evaluation.py. Firstly it loads pre-trained embedding matrices and their associated word indices of the previously trained models. Then it utilizes a methodical approach to identify and adjust biased vector representations through the application of linear projections along predefined informative dimensions. The script then recalculates embedding vectors to align with a neutral direction, effectively reducing bias. The "debiased matrices" are saved and then the model can be re-trained on this embedding matrix in order to perform bias_evaluation again afterwards, where the results can then be compared with the bias results before applying our bias mitigation technique. The results are documented in detail in our project documentation. 


