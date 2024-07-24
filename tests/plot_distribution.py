import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data
df_flocking = pd.read_csv('stats/go_to_position/evaluation_9/mean.csv')

# Plot Reward Distribution Histogram
plt.figure(figsize=(12, 6))
sns.histplot(df_flocking['Reward'], bins=10, kde=True)
plt.xlabel('Reward')
plt.ylabel('Frequency')
plt.title('Reward Distribution Histogram')
plt.grid(True)
plt.show()

# Data
df_flocking = pd.read_csv('stats/flocking/evaluation_9/mean.csv')

# Plot Reward Distribution Histogram
plt.figure(figsize=(12, 6))
sns.histplot(df_flocking['Reward'], bins=10, kde=True)
plt.xlabel('Reward')
plt.ylabel('Frequency')
plt.title('Reward Distribution Histogram')
plt.grid(True)
plt.show()


