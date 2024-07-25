import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df_flocking_goto = pd.read_csv('stats/go_to_position/evaluation_9/mean.csv')
df_flocking_flock = pd.read_csv('stats/flocking/evaluation_9/mean.csv')
df_flocking_goto_12 = pd.read_csv('stats/go_to_position/evaluation_12/mean.csv')
df_flocking_flock_12 = pd.read_csv('stats/flocking/evaluation_12/mean.csv')

sns.set_theme(style="dark")

plt.figure(figsize=(8, 6))
sns.histplot(df_flocking_goto['Reward'], bins=10, kde=True, color='green')
plt.xlabel('Reward')
plt.ylabel('Frequency')
plt.title('Reward Distribution Histogram for Go To Position - Evaluation 9')
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
sns.histplot(df_flocking_flock['Reward'], bins=10, kde=True, color='blue')
plt.xlabel('Reward')
plt.ylabel('Frequency')
plt.title('Reward Distribution Histogram for Flocking - Evaluation 9')
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
sns.histplot(df_flocking_goto_12['Reward'], bins=10, kde=True, color='green')
plt.xlabel('Reward')
plt.ylabel('Frequency')
plt.title('Reward Distribution Histogram for Go To Position - Evaluation 12')
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
sns.histplot(df_flocking_flock_12['Reward'], bins=10, kde=True, color='blue')
plt.xlabel('Reward')
plt.ylabel('Frequency')
plt.title('Reward Distribution Histogram for Flocking - Evaluation 12')
plt.grid(True)
plt.show()
