from curses.ascii import SI
import sys
import os
scenarios_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scenarios'))
sys.path.insert(0, scenarios_dir)
import csv
import sys
import os
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from vmas import make_env
from go_to_position_scenario import GoToPositionScenario
from cohesion_scenario import CohesionScenario
from flocking_scenario import FlockingScenario
from obstacle_avoidance_scenario import ObstacleAvoidanceScenario
from torch_geometric.data import Data, Batch
import torch.utils.tensorboard as tensorboard
import random
import torch.nn.utils as utils
import torch.nn as nn

class GraphReplayBuffer:

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, graph_observation, actions, rewards, next_graph_observation):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (graph_observation, actions, rewards, next_graph_observation)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        sample = random.sample(self.buffer, batch_size)
        observations = [s[0] for s in sample]
        actions = [s[1] for s in sample]
        rewards = [s[2] for s in sample]
        next_graph_observations = [s[3] for s in sample]
        return (Batch.from_data_list(observations), torch.cat(actions), torch.cat(rewards), Batch.from_data_list(next_graph_observations))

    def __len__(self):
        return len(self.buffer)

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, add_self_loops=False, bias=True)
        self.conv2 = GATConv(hidden_dim, hidden_dim, add_self_loops=False, bias=True)
        self.conv3 = GATConv(hidden_dim, hidden_dim, add_self_loops=False, bias=True)
        self.lin1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)
        x = torch.relu(x)
        x = self.lin1(x)
        x = torch.relu(x)
        x = self.lin2(x)
        return x

class DQNTrainer:
    def __init__(self, env):
        self.env = env
        self.n_input = self.env.observation_space['agent0'].shape[0] + 1
        self.n_output = env.action_space['agent0'].n
        self.model = GCN(input_dim=self.n_input, hidden_dim=32, output_dim=self.n_output)
        self.target_model = GCN(input_dim=self.n_input, hidden_dim=32, output_dim=self.n_output)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=0.0001)
        self.replay_buffer = GraphReplayBuffer(6000)
        self.writer = tensorboard.SummaryWriter()
        self.episode_rewards = []
        self.episode_losses = []
        self.episode_obstacle_hits = []
        self.rewards_buffer = []
        self.obstacle_hits_buffer = []

    def create_graph_from_observations(self, observations):
        node_features = [observations[f'agent{i}'] for i in range(len(observations))]
        node_features = torch.stack(node_features, dim=0).squeeze(dim=1)
        
        agent_ids = torch.arange(len(observations)).float().unsqueeze(1)
        node_features = torch.cat([node_features, agent_ids], dim=1)
        
        num_agents = self.env.n_agents
        edge_index = []
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                edge_index.append([i, j])
                edge_index.append([j, i])
        edge_index.append([0,0])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        graph_data = Data(x=node_features, edge_index=edge_index)
        return graph_data
    
    def train_step_dqn(self, batch_size, model, target_model, ticks, gamma=0.99, update_target_every=10):
        if len(self.replay_buffer) < batch_size:
            return 0
        model.train()
        self.optimizer.zero_grad()
        obs, actions, rewards, nextObs = self.replay_buffer.sample(batch_size)

        values = model(obs).gather(1, actions.unsqueeze(1))
        nextValues = target_model(nextObs).max(dim=1)[0].detach()
        targetValues = rewards + gamma * nextValues
        loss = nn.SmoothL1Loss()(values, targetValues.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 1)
        self.optimizer.step()
            
        if ticks % update_target_every == 0:
            target_model.load_state_dict(model.state_dict())
        self.writer.add_scalar('Loss', loss.item(), ticks)
        return loss.item()

    def train_model(self, config):
        model_name = config["model_name"]
        epsilon = config["epsilon"]
        epsilon_decay = config["epsilon_decay"]
        min_epsilon = config["min_epsilon"]
        episodes = config["episodes"]
        ticks = 0

        for episode in range(episodes):  
            observations = self.env.reset()    
            episode_loss = 0
            total_episode_reward = torch.zeros(self.env.n_agents)

            for _ in range(self.env.max_steps):
                if episode % 100 == 0:
                    self.env.render(
                        mode="rgb_array",
                        agent_index_focus=None,
                        visualize_when_rgb=True,
                    )
                ticks += 1
                graph_data = self.create_graph_from_observations(observations)
                self.model.eval()
                logits = self.model(graph_data)

                if random.random() < epsilon:
                    actions = torch.tensor([random.randint(0, 8) for _ in range(len(self.env.agents))])
                else:
                    actions = torch.argmax(logits, dim=1)
                actions_dict = {f'agent{i}': torch.tensor([actions[i].item()]) for i in range(len(self.env.agents))}
                newObservations, rewards, done, _ = self.env.step(actions_dict)

                rewards_tensor = torch.tensor([rewards[f'agent{i}'] for i in range(len(self.env.agents))], dtype=torch.float)
                self.replay_buffer.push(graph_data, actions, rewards_tensor, self.create_graph_from_observations(newObservations))
                
                self.writer.add_scalar('Reward', rewards_tensor.sum().item(), ticks)
                loss = self.train_step_dqn(128, self.model, self.target_model, ticks, update_target_every=10)
                episode_loss += loss
                total_episode_reward += rewards_tensor
                observations = newObservations

            epsilon = max(min_epsilon, epsilon * epsilon_decay)
            
            average_loss = episode_loss / self.env.max_steps
            self.episode_losses.append(average_loss)
            self.rewards_buffer.append(total_episode_reward[0])

            if (episode + 1) % 10 == 0:
                mean_reward = sum(self.rewards_buffer) / 10
                self.episode_rewards.append(mean_reward)
                self.rewards_buffer = []

                mean_hits = sum(self.obstacle_hits_buffer) / 10
                self.episode_obstacle_hits.append(mean_hits)
                self.obstacle_hits_buffer = []
                """ # Run evaluation after every 10 episodes of training
                eval_reward = self.evaluate_policy(10)
                self.episode_rewards.append(eval_reward) """

            print(f'Episode {episode}, Loss: {average_loss}, Reward: {total_episode_reward.sum().item()}, Epsilon: {epsilon}')

        print("Training completed")
        torch.save(self.model.state_dict(), model_name + '.pth')
        print("Model saved successfully!")

        self.save_metrics_to_csv()

    def evaluate_policy(self, eval_episodes):
        total_eval_reward = 0
        for _ in range(eval_episodes):
            observations = self.env.reset()
            episode_reward = torch.zeros(self.env.n_agents)
            for _ in range(self.env.max_steps):
                graph_data = self.create_graph_from_observations(observations)
                self.model.eval()
                with torch.no_grad():
                    logits = self.model(graph_data)
                actions = torch.argmax(logits, dim=1)
                actions_dict = {f'agent{i}': torch.tensor([actions[i].item()]) for i in range(len(self.env.agents))}
                newObservations, rewards, done, _ = self.env.step(actions_dict)
                rewards_tensor = torch.tensor([rewards[f'agent{i}'] for i in range(len(self.env.agents))], dtype=torch.float)
                episode_reward += rewards_tensor
                observations = newObservations
            total_eval_reward += episode_reward[0]
        return total_eval_reward / eval_episodes

    def save_metrics_to_csv(self):
        with open('obstacle_avoidance_stats_4842.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Episode', 'Reward'])
            for i in range(len(self.episode_losses)):
                if (i + 1) % 10 == 0:
                    writer.writerow([i, self.episode_rewards[i // 10].item()])

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":

    SEED = 4842

    set_seed(SEED)

    env = make_env(
        scenario=FlockingScenario(),
        num_envs=1,
        device="cpu",
        continuous_actions=False,
        wrapper=None,
        max_steps=100,
        dict_spaces=True,
        n_agents=12,
        seed=SEED
    )

    config = {
        'model_name': 'flocking_model_12',
        'epsilon': 0.99,
        'epsilon_decay' : 0.9,
        'min_epsilon' : 0.05,
        'episodes' : 800
    }
    
    trainer = DQNTrainer(env)
    trainer.train_model(config)