import pandas as pd

synonyms = list(zip(pd.read_csv('../data/synonyms.csv')['Target'], pd.read_csv('../data/synonyms.csv')['AugmentedSentence']))
backtranslation = list(zip(pd.read_csv('../data/backtranslation.csv')['target'], pd.read_csv('../data/backtranslation.csv')['comment_text']))

# I have to convert titles into same header


# stack both lists
all_augmented = synonyms + backtranslation
# save stacked list to a csv file
pd.DataFrame(all_augmented).to_csv('../data/all_augmented.csv', index=False)

# search for missing values in all_augmented
all_augmented = pd.read_csv('../data/all_augmented.csv')
nulls = all_augmented.isnull().sum()

# drop missing values
all_augmented = all_augmented.dropna()

# save cleaned data to a csv file
all_augmented.to_csv('../data/all_augmented.csv', index=False)