import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from vmas import make_env
from custom_scenario import CustomScenario
from torch_geometric.data import Data, Batch
import torch.utils.tensorboard as tensorboard
import random
import torch.nn.utils as utils
import torch.nn as nn
env = make_env(
    scenario=CustomScenario(),
    num_envs=1,
    device="cpu",
    continuous_actions=False,
    wrapper=None,
    max_steps=200,
    dict_spaces=True,
    n_agents=9,
)

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

writer = tensorboard.SummaryWriter()

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

def create_graph_from_observations(observations):
    node_features = [observations[f'agent{i}'] for i in range(len(observations))]
    node_features = torch.stack(node_features, dim=0).squeeze(dim=1)
    
    # Aggiunge un identificatore unico per ogni agente
    agent_ids = torch.arange(len(observations)).float().unsqueeze(1)
    
    node_features = torch.cat([node_features, agent_ids], dim=1)

    # DEBUG: osservazioni prese come input e node_features create
    #print(observations)
    #print(node_features.shape)
    
    num_agents = node_features.size(0)
    edge_index = []
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            edge_index.append([i, j])
            edge_index.append([j, i])
    edge_index.append([0,0])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    graph_data = Data(x=node_features, edge_index=edge_index)
    return graph_data


def train_model():
    num_actions = 9  
    model = GCN(input_dim=5, hidden_dim=32, output_dim=num_actions) 
    target_model = GCN(input_dim=5, hidden_dim=32, output_dim=num_actions)
    target_model.load_state_dict(model.state_dict())
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0001)
    replay = GraphReplayBuffer(6000)

    def train_step_dqn(replay_buffer, batch_size, model, target_model, ticks, gamma=0.99, update_target_every=10):
        if(len(replay_buffer.buffer) < batch_size):
            return 0
        model.train()
        optimizer.zero_grad()
        (obs, actions, rewards, nextObs) = replay_buffer.sample(batch_size)
        #rewards = torch.nn.functional.normalize(rewards, dim=0)
        values = model(obs).gather(1, actions.unsqueeze(1))
        nextValues = target_model(nextObs).max(dim=1)[0].detach()
        targetValues = rewards + gamma * nextValues
        loss = nn.SmoothL1Loss()(values, targetValues.unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 1)
        optimizer.step()
            # 7. Update Target Network
        if ticks % update_target_every == 0:
            target_model.load_state_dict(model.state_dict())
        writer.add_scalar('Loss', loss.item(), ticks)
        return loss.item()
    
    epsilon = 0.99
    epsilon_decay = 0.9
    min_epsilon = 0.05
    ticks = 0
    episodes = 800
    for episode in range(episodes):  
        observations = env.reset()    
        episode_loss = 0
        total_episode_reward = torch.zeros(env.n_agents)  
        for step in range(100):
            if episode % 10 == 0:
                env.render(
                    mode="rgb_array",
                    agent_index_focus=None,
                    visualize_when_rgb=True,
                )
            ticks += 1
            graph_data = create_graph_from_observations(observations)
            model.eval()
            logits = model(graph_data)
            # DEBUG: logits restituiti in output dalla GCN
            #print(f'Logits shape: {logits.shape}')  

            if random.random() < epsilon:
                actions = torch.tensor([random.randint(0, num_actions - 1) for _ in range(len(env.agents))])
            else:
                actions = torch.argmax(logits, dim=1)
            actions_dict = {f'agent{i}': torch.tensor([actions[i].item()]) for i in range(len(env.agents))}
            newObservations, rewards, done, _ = env.step(actions_dict)

            rewards_tensor = torch.tensor([rewards[f'agent{i}'] for i in range(len(env.agents))], dtype=torch.float)
            replay.push(graph_data, actions, rewards_tensor, create_graph_from_observations(newObservations))
            
            writer.add_scalar('Reward', rewards_tensor.sum().item(), ticks)
            loss = train_step_dqn(replay, 128, model, target_model, ticks, update_target_every=10)
            episode_loss += loss
            total_episode_reward += rewards_tensor
            observations = newObservations

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        average_loss = episode_loss / 100
        print(f'Episode {episode}, Loss: {average_loss}, Reward: {total_episode_reward}, Epsilon: {epsilon}')

    print("Training completed")
    torch.save(model.state_dict(), 'cohesion_collision.pth')
    print("Model saved successfully!")

if __name__ == "__main__":
    train_model()
