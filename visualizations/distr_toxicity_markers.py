import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt

filepath = "../data/train.csv"
data = pd.read_csv(filepath)

# Calculate the sum of toxicity marks per comment
data['total_toxicity'] = data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].sum(axis=1)

# Print the counts of total toxicity marks
print(data['total_toxicity'].value_counts())

# Plotting
sns.countplot(x='total_toxicity', data=data)
plt.title('Distribution of Total Toxicity Marks per Comment', fontsize=18, fontweight='bold')
plt.xlabel('Total Toxicity Marks', fontsize=16)
plt.ylabel('Number of Comments', fontsize=16)

# Increasing the size of x and y ticks
plt.tick_params(axis='both', which='major', labelsize=14)

# Save the figure
plt.savefig('figures/toxicity_distribution.png')

plt.show()