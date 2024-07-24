import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data
df_flocking_goto = pd.read_csv('stats/go_to_position/evaluation_9/mean.csv')
df_flocking_flock = pd.read_csv('stats/flocking/evaluation_9/mean.csv')
df_flocking_goto_12 = pd.read_csv('stats/go_to_position/evaluation_12/mean.csv')
df_flocking_flock_12 = pd.read_csv('stats/flocking/evaluation_12/mean.csv')

# Set Seaborn theme to dark
sns.set_theme(style="dark")

# Create a figure with two subplots side by side
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot Reward Distribution Histogram for the first dataset (green)
sns.histplot(df_flocking_goto['Reward'], bins=10, kde=True, color='green', ax=axes[0])
axes[0].lines[0].set_color('green')  # Set the KDE line color to green
axes[0].set_xlabel('Reward')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Reward Distribution Histogram for Go To Position')
axes[0].grid(True) 

# Plot Reward Distribution Histogram for the second dataset (blue)
sns.histplot(df_flocking_flock['Reward'], bins=10, kde=True, color='blue', ax=axes[1])
axes[1].lines[0].set_color('blue')  # Set the KDE line color to blue
axes[1].set_xlabel('Reward')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Reward Distribution Histogram for Flocking')
axes[1].grid(True) 

# Adjust layout
plt.tight_layout()
plt.show()

# Set Seaborn theme to dark
sns.set_theme(style="dark")

# Create a figure with two subplots side by side
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot Reward Distribution Histogram for the first dataset (green)
sns.histplot(df_flocking_goto_12['Reward'], bins=10, kde=True, color='green', ax=axes[0])
axes[0].lines[0].set_color('green')  # Set the KDE line color to green
axes[0].set_xlabel('Reward')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Reward Distribution Histogram for Go To Position')
axes[0].grid(True) 

# Plot Reward Distribution Histogram for the second dataset (blue)
sns.histplot(df_flocking_flock_12['Reward'], bins=10, kde=True, color='blue', ax=axes[1])
axes[1].lines[0].set_color('blue')  # Set the KDE line color to blue
axes[1].set_xlabel('Reward')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Reward Distribution Histogram for Flocking')
axes[1].grid(True) 

# Adjust layout
plt.tight_layout()
plt.show()
