import torch
from torch_geometric.data import Data
from train_gcn_dqn import DQNTrainer
import os
import time
import csv


def create_graph_from_observations(self, observations, num_agents):
    node_features = [observations[f'agent{i}'] for i in range(len(observations))]
    node_features = torch.stack(node_features, dim=0).squeeze(dim=1)

    agent_ids = torch.arange(len(observations)).float().unsqueeze(1)
    node_features = torch.cat([node_features, agent_ids], dim=1)
    edge_index = []
    for i in range(num_agents):
        # take the 3 nearest agents
        distance_to_i = torch.linalg.norm(node_features[:, :2] - node_features[i, :2], dim=1)
        _, nearest_agents = torch.topk(distance_to_i, 10, largest=False)
        for agent in nearest_agents:
            edge_index.append([i, agent.item()])
            edge_index.append([agent.item(), i])
    edge_index.append([0, 0])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    graph_data = Data(x=node_features, edge_index=edge_index)
    return graph_data

class Simulator:
    def __init__(self, env, model, episodes, env_name, seed, output_dir='test_stats/', render=False):
        self.env = env
        self.model = model
        self.episode_rewards = []
        self.distance_at_the_end = []
        self.distance_at_the_beginning = []
        self.total_collisions = []
        self.rewards_buffer = []
        self.episodes = episodes
        self.env_name = env_name
        self.seed = seed
        self.output_dir = output_dir
        self.render = render
        self.all_positions_x = []
        self.all_positions_y = []
        self.all_distances = []
        self.all_hits = []

    def run_simulation(self):

        for episode in range(self.episodes):
            total_episode_reward = torch.zeros(self.env.n_agents)
            observations = self.env.reset()
            init_time = time.time()
            total_reward = 0
            collision_in_episode = 0
            all_position_x_in_episode = []
            all_position_y_in_episode = []
            all_distances_in_episode = []
            all_hits_in_episode = []
            for i in range(self.env.max_steps):
                graph_data = create_graph_from_observations(self, observations, self.env.n_agents)

                with torch.no_grad():
                    logits = self.model(graph_data)                    
                    actions = torch.argmax(logits, dim=1)
                
                actions_dict = {f'agent{i}': torch.tensor([actions[i].item()]) for i in range(len(self.env.agents))}
                
                observations, rewards, _ , _ = self.env.step(actions_dict)
                if i == 0:
                    self.distance_at_the_beginning.append(self.env.scenario.average_distance_to_goal())
                rewards_tensor = torch.tensor([rewards[f'agent{i}'] for i in range(len(self.env.agents))], dtype=torch.float)

                total_episode_reward += rewards_tensor

                total_reward += sum(rewards.values())

                collision_in_episode += self.env.scenario.obstacles_hits()

                # for each agent, take the x and y
                agent_positions = [observations[f'agent{i}'][:, :2] for i in range(len(observations))]
                all_position_x = [agent[0][0].item() for agent in agent_positions]
                all_position_y = [agent[0][1].item() for agent in agent_positions]
                all_position_x_in_episode.append(all_position_x)
                all_position_y_in_episode.append(all_position_y)

                all_distances_in_episode.append(self.env.scenario.average_distance_to_goal().item())
                all_hits_in_episode.append(self.env.scenario.obstacles_hits().item())
                if self.render:
                    frame = self.env.render(
                        mode="rgb_array",
                        agent_index_focus=None,
                        visualize_when_rgb=True,
                    )

            self.all_positions_x.append(all_position_x_in_episode)
            self.all_positions_y.append(all_position_y_in_episode)
            self.all_distances.append(all_distances_in_episode)
            self.all_hits.append(all_hits_in_episode)

            total_time = time.time() - init_time
            print(
                f"It took: {total_time}s for {self.env.max_steps} steps of episode {episode} with {total_reward} total reward, on device {self.env.device} "
                f"for test_gcn_vmas scenario."
            )
            averaged_global_reward = total_reward / self.env.max_steps
            self.total_collisions.append(collision_in_episode)
            self.distance_at_the_end.append(self.env.scenario.average_distance_to_goal())
            self.episode_rewards.append(averaged_global_reward.item())
        self.save_metrics_to_csv()

    def save_metrics_to_csv(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        where = self.output_dir
        file_name = f'{where}{self.env_name}_seed_{self.seed}.csv'
        with open(where + "/result.csv", mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Episode', 'Reward', 'Collisions', 'Distance (end)', 'Distance (beginning)'])
            for i in range(self.episodes):
                writer.writerow([
                    i,
                    self.episode_rewards[i],
                    self.total_collisions[i].item(),
                    self.distance_at_the_end[i].item(),
                    self.distance_at_the_beginning[i].item()
                ])

        folder_positions = f'{where}/positions'
        if not os.path.exists(folder_positions):
            os.makedirs(folder_positions)

        for i in range(len(self.all_positions_x)):
            with open(f'{folder_positions}/positions_episode_{i}_x.csv', mode='w', newline='') as file:
                writer = csv.writer(file)
                xs_title = [f'X{i}' for i in range(len(self.all_positions_x[i][0]))]
                row = ['Tick'] + xs_title
                writer.writerow(row)
                for j in range(len(self.all_positions_x[i])):
                    row = [j] + self.all_positions_x[i][j]
                    writer.writerow(row)

        for i in range(len(self.all_positions_y)):
            with open(f'{folder_positions}/positions_episode_{i}_y.csv', mode='w', newline='') as file:
                writer = csv.writer(file)
                ys_title = [f'Y{i}' for i in range(len(self.all_positions_y[i][0]))]
                row = ['Tick'] + ys_title
                writer.writerow(row)
                for j in range(len(self.all_positions_y[i])):
                    row = [j] + self.all_positions_y[i][j]
                    writer.writerow(row)

        # Save all distances and all hits

        file_data = f'{where}/data'
        if not os.path.exists(file_data):
            os.makedirs(file_data)

        print(len(self.all_distances[0]))
        for i in range(len(self.all_distances)):
            with open(f'{file_data}/distances_episode_{i}.csv', mode='w', newline='') as file:
                writer = csv.writer(file)
                row = ['Tick', 'Distance', 'Hits']
                writer.writerow(row)
                for j in range(len(self.all_distances[i])):
                    row = [j, self.all_distances[i][j], self.all_hits[i][j]]
                    writer.writerow(row)
