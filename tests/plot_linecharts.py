import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'stats/go_to_position/training/mean.csv'
data = pd.read_csv(file_path)

sns.set_theme(style="dark")
plt.figure(figsize=(8, 6))
sns.lineplot(x='Episode', y='Reward', data=data, label='Reward', color='g')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Go To Position Rewards')
plt.legend()
plt.grid(True)
plt.show()

file_path = 'stats/flocking/training/mean.csv'
data = pd.read_csv(file_path)

sns.set_theme(style="dark")
plt.figure(figsize=(8, 6))
plt.plot(data['Episode'], data['Reward'], label='Reward', color='b')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Flocking Rewards')
plt.legend()
plt.grid(True)
plt.show()

file_path = 'stats/obstacle_avoidance/eval_mean.csv'
data = pd.read_csv(file_path)

sns.set_theme(style="dark")
plt.figure(figsize=(8, 6))
plt.plot(data['Episode'], data['Reward'], label='Reward', color='r')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Obstacle Avoidance Rewards')
plt.legend()
plt.grid(True)
plt.show()

file_path = 'stats/obstacle_avoidance/hits_mean.csv'
data = pd.read_csv(file_path)

sns.set_theme(style="dark")
plt.figure(figsize=(8, 6))
plt.plot(data['Episode'], data['Hits'], label='Hits', color='g')
plt.xlabel('Episode')
plt.ylabel('Hits')
plt.title('Obstacle Hits')
plt.legend()
plt.grid(True)
plt.show()