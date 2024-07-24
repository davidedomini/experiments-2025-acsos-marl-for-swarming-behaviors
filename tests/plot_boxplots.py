import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load data
df_goto_position = pd.read_csv('stats/go_to_position/evaluation_5/mean.csv')
df_flocking = pd.read_csv('stats/flocking/evaluation_5/mean.csv')

# Set Seaborn theme for aesthetics
sns.set_theme(style="whitegrid")

# Define a color palette
palette = sns.color_palette("Set2")

# Create boxplot for Go To Position Simulation
plt.figure(figsize=(8, 6))
sns.boxplot(y=df_goto_position['Mean'], color=palette[0], 
            boxprops=dict(linewidth=1.5),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
            medianprops=dict(linewidth=1.5),
            flierprops=dict(markerfacecolor=palette[0], marker='o', markersize=8, linestyle='none'))
plt.title('Go To Position Simulation', fontsize=16)
plt.ylabel('Rewards', fontsize=14)
plt.yticks(fontsize=12)
plt.ylim(df_goto_position['Mean'].min() - 5, df_goto_position['Mean'].max() + 5)  # Adjust limits
plt.show()

# Create boxplot for Flocking Simulation
plt.figure(figsize=(8, 6))
sns.boxplot(y=df_flocking['Mean'], color=palette[1], 
            boxprops=dict(linewidth=1.5),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
            medianprops=dict(linewidth=1.5),
            flierprops=dict(markerfacecolor=palette[1], marker='o', markersize=8, linestyle='none'))
plt.title('Flocking Simulation', fontsize=16)
plt.ylabel('Rewards', fontsize=14)
plt.yticks(fontsize=12)
plt.ylim(df_flocking['Mean'].min() - 5, df_flocking['Mean'].max() + 5)  # Adjust limits
plt.show()
