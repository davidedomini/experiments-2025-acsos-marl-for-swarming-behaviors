import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load Data from CSV
df = pd.read_csv('output.csv')

# Create Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(y=df['Mean'])
plt.title('Boxplot of Means')
plt.ylabel('Mean')
plt.show()
