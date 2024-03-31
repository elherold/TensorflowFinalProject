import pandas as pd
import re
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
    return input_text.strip()

def main():
    """
    Main function to execute the script. It loads the dataset, preprocesses the texts,
    creates lists of zero and nonzero targets, and saves them to CSV files.
    """
    # Load the dataset
    filepath = '../data/train.csv'
    filepath = '../data/backtranslation_augmented.csv'
    data = pd.read_csv(filepath)

    texts = data['comment_text'].astype(str)
    targets = data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values

    # Download stopwords
    #nltk.download('stopwords')

    # Preprocess texts
    X = [preprocess_text(s) for s in texts]

    # Create zero and nonzero targets lists
    zero_targets = []
    nonzero_targets = []

    paired_data = list(zip(targets, X))
    print(f"length of paired_data: {len(paired_data)}")
    print(f"Head of paired data: {paired_data[:5]}")

    for idx, (target, sentence) in enumerate(paired_data):
        if all(x == 0 for x in target):
            zero_targets.append((0, sentence))
        else:
            nonzero_targets.append((1, sentence))
        # Now using idx to check every 4 iterations
        if idx % 4 == 0:
            print(f"Processed {idx} datapoints")


    # Save to CSV files
    # Check if the csv files already exist
            
    filepath_one = '../data/zero_targets.csv'
    #filepath_two = '../data/nonzero_targets.csv'
    filepath_two = '../data/backtranslation.csv'

    if not os.path.exists(filepath_one):
        pd.DataFrame(zero_targets, columns=['target', 'comment_text']).to_csv(filepath_one, index=False)
        print(f"File saved at {filepath_one}")
    else: 
        print(f"File already exists at {filepath_one}")

    if not os.path.exists(filepath_two):
        pd.DataFrame(nonzero_targets, columns=['target', 'comment_text']).to_csv(filepath_two, index=False)
        print(f"File saved at {filepath_two}")
    else:
        print(f"File already exists at {filepath_two}")


if __name__ == '__main__':
    main()
