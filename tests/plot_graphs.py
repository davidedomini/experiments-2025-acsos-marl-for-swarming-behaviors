import pandas as pd
import matplotlib.pyplot as plt

file_path = 'training_metrics_go_to_position.csv'
data = pd.read_csv(file_path)

# Plot the trend of the reward
plt.figure(figsize=(12, 6))
plt.plot(data['Episode'], data['Reward'], label='Reward', color='b')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward Trend Over Episodes')
plt.legend()
plt.grid(True)
plt.show()

# Plot the trend of the loss
plt.figure(figsize=(12, 6))
plt.plot(data['Episode'], data['Loss'], label='Loss', color='r')
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.title('Loss Trend Over Episodes')
plt.legend()
plt.grid(True)
plt.show()
