import pandas as pd
import matplotlib.pyplot as plt

file_path = 'training_metrics_go_to_position_eval_normalized.csv'
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