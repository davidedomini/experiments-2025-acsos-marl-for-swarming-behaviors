import pandas as pd
import matplotlib.pyplot as plt

file_path = 'hits_stats_4842.csv'
data = pd.read_csv(file_path)

# Plot the trend of the reward
plt.figure(figsize=(12, 6))
plt.plot(data['Episode'], data['Hits'], label='Hits', color='b')
plt.xlabel('Episode')
plt.ylabel('Hits')
plt.title('Hits Trend Over Episodes')
plt.legend()
plt.grid(True)
plt.show()

""" file_path = 'obstacle_avoidance_stats_4842.csv'
data = pd.read_csv(file_path)

# Plot the trend of the reward mean
plt.figure(figsize=(12, 6))
plt.plot(data['Episode'], data['Reward'], label='Reward', color='b')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward Mean Trend Over Episodes')
plt.legend()
plt.grid(True)
plt.show() """