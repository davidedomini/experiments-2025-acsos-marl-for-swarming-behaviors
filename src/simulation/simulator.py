import torch
from train_gcn_dqn import DQNTrainer
import time
import csv

class Simulator:
    def __init__(self, env, model):
        self.env = env
        self.model = model
        self.trainer = DQNTrainer(self.env)
        self.episode_rewards = []
        self.rewards_buffer = []

    def run_simulation(self):

        for episode in range(400):
            total_episode_reward = torch.zeros(self.env.n_agents)
            observations = self.env.reset()
            init_time = time.time()
            total_reward = 0

            for _ in range(self.env.max_steps):
                graph_data = self.trainer.create_graph_from_observations(observations)

                with torch.no_grad():
                    logits = self.model(graph_data)                    
                    actions = torch.argmax(logits, dim=1)
                
                actions_dict = {f'agent{i}': torch.tensor([actions[i].item()]) for i in range(len(self.env.agents))}
                
                observations, rewards, _ , _ = self.env.step(actions_dict)

                rewards_tensor = torch.tensor([rewards[f'agent{i}'] for i in range(len(self.env.agents))], dtype=torch.float)

                total_episode_reward += rewards_tensor

                total_reward += sum(rewards.values())

                frame = self.env.render(
                    mode="rgb_array",
                    agent_index_focus=None,
                    visualize_when_rgb=True,
                )

            total_time = time.time() - init_time
            print(
                f"It took: {total_time}s for {self.env.max_steps} steps of episode {episode} with {total_reward} total reward, on device {self.env.device} "
                f"for test_gcn_vmas scenario."
            )

            self.rewards_buffer.append(total_episode_reward[0])
            if (episode + 1) % 10 == 0:
                mean_reward = sum(self.rewards_buffer) / 10
                self.episode_rewards.append(mean_reward)
                self.rewards_buffer = []

        self.save_metrics_to_csv()

    def save_metrics_to_csv(self):
        with open('go_to_position_stats_eval_8140.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Episode', 'Reward'])
            for i in range(400):
                if (i + 1) % 10 == 0:
                    writer.writerow([i, self.episode_rewards[i // 10].item()])

