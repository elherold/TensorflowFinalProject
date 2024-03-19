import pandas as pd
import matplotlib.pyplot as plt

# Load the data
filepath = "../data/train.csv"
data = pd.read_csv(filepath)

# Define the categories
categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Calculate counts
all_zero_count = len(data[(data[categories] == 0).all(axis=1)])
at_least_one_count = len(data) - all_zero_count

# Create a DataFrame for plotting
counts_df = pd.DataFrame({'Type': ['All Zero', 'At Least One 1'], 'Counts': [all_zero_count, at_least_one_count]})

# Plotting
plt.figure(figsize=(8, 6))
bar_plot = plt.bar(counts_df['Type'], counts_df['Counts'])

plt.title('Comparison of Comment Types', fontsize=24, fontweight='bold')
plt.xlabel('Type of Comment', fontsize=18)
plt.ylabel('Number of Comments', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Save the figure
plt.savefig('figures/comment_type_comparison.png')

plt.show()
