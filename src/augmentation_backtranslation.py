import pandas as pd
import numpy as np
import os

# Import libraries for translation (assuming you have them installed)
from google.cloud import translate_v2 as translate


def translate_text(target: str, text: str, client) -> dict:
    """Translates text into the target language.

    Target must be an ISO 639-1 language code.
    See https://g.co/cloud/translate/v2/translate-reference#supported_languages
    """

    if isinstance(text, bytes):
        text = text.decode("utf-8")

    # API CALL
    result = client.translate(text, target_language=target)

    # print("Text: {}".format(result["input"]))
    # print("Translation: {}".format(result["translatedText"]))
    # print("Detected source language: {}".format(result["detectedSourceLanguage"]))

    return result


def augment_text_translation(df):
    """
    This function performs text augmentation using translation.

    Args:
        df (pandas.DataFrame): The DataFrame containing comments.

    Returns:
        list, list: A list of back-translated comments and a list of translated comments.
    """

    comments = df[df["toxic"] == 1]["comment_text"].tolist()
    translate_client = translate.Client()
    translated_comments = [
        translate_text("fr", comment, translate_client)["translatedText"]
        for comment in comments
    ]
    print("first translation finished")
    back_translated = [
        translate_text("en", comment, translate_client)["translatedText"]
        for comment in translated_comments
    ]

    return back_translated, translated_comments


def augmentation_backtranslation():
    """
    This function performs text augmentation using Googles translation API and saves the back-translated data.
    Make sure to have a valid key stored in ../data/api_key.json.
    You can create a key at https://cloud.google.com/translate/docs/setup.
    """
    # Define file paths (replace with your actual paths)
    filepath_train = "datasets/ruddit/train.csv"
    error

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../data/api_key.json"

    # Load data
    df_train = pd.read_csv(filepath_train)

    # Filter data
    df_train_1 = df_train[
        df_train[
            ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        ].sum(axis=1)
        > 0
    ]
    df_train_1_short = df_train_1[df_train_1["comment_text"].str.len() < 1200]
    print("finished preloading")
    # Text Augmentation using translation
    df_todo = df_train_1_short.copy()  # Avoid modifying original DataFrame
    back_translated, translated_comments = augment_text_translation(df_todo)

    # Save back-translated data (assuming you have write permissions)
    back_translated_df = pd.DataFrame({"comment_text": back_translated})
    back_translated_df.to_csv("data/backtranslation_new.csv", index=False)


if __name__ == "__main__":
    augmentation_backtranslation()
