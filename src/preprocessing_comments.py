import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import os


def remove_tags(text):
    """
    Remove HTML tags from a text string.

    Args:
        text (str): A string containing HTML tags.

    Returns:
        str: The text string with HTML tags removed.
    """
    TAG_RE = re.compile(r'<[^>]+>')
    return TAG_RE.sub('', text)

print("Loading data...Test")

def preprocess_text(input_text):
    """
    Preprocess the input text by converting to lowercase, removing HTML tags,
    punctuations, numbers, single characters, multiple spaces, and stopwords.

    Args:
        input_text (str): The text to be preprocessed.

    Returns:
        str: The preprocessed text.
    """
    # Convert to lowercase
    input_text = input_text.lower()
    # Remove HTML tags
    input_text = remove_tags(input_text)
    # Remove punctuations and numbers
    input_text = re.sub('[^a-zA-Z]', ' ', input_text)
    # Remove single characters and multiple spaces
    input_text = re.sub(r'\s+[a-zA-Z]\s', ' ', input_text)
    input_text = re.sub(r'\s+', ' ', input_text)
    # Remove stopwords
    pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
    input_text = pattern.sub('', input_text)
    return input_text.strip()

def main():
    """
    Main function to execute the script. It loads the dataset, preprocesses the texts,
    creates lists of zero and nonzero targets, and saves them to CSV files.
    """
    # Load the dataset
    filepath = 'data/train.csv'
    data = pd.read_csv(filepath)

    texts = data['comment_text'].astype(str)
    targets = data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values

    # Download stopwords
    nltk.download('stopwords')

    # Preprocess texts
    X = [preprocess_text(s) for s in texts]

    # Create zero and nonzero targets lists
    zero_targets = []
    nonzero_targets = []

    paired_data = list(zip(targets, X))

    for target, sentence in paired_data:
        if all(x == 0 for x in target):
            zero_targets.append((0, sentence))
        else:
            nonzero_targets.append((1, sentence))
        if len(paired_data) % 4 == 0:
            print(f"Processed {len(zero_targets) + len(nonzero_targets)} datapoints")


    # Save to CSV files
    # Check if the csv files already exist
    if not os.path.exists('../data/zero_targets.csv'):
        pd.DataFrame(zero_targets, columns=['target', 'comment_text']).to_csv('../data/zero_targets.csv', index=False)
        print("zero_targets.csv saved")
    else: 
        print("zero_targets.csv already exists")

    if not os.path.exists('../data/nonzero_targets.csv'):
        pd.DataFrame(nonzero_targets, columns=['target', 'comment_text']).to_csv('../data/nonzero_targets.csv', index=False)
        print("nonzero_targets.csv saved")
    else:
        print("nonzero_targets.csv already exists")


if __name__ == '__main__':
    main()
