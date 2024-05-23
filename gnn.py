import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from vmas import make_env, Wrapper
from custom_scenario import CustomScenario
from torch_geometric.data import Data


env = make_env(
        scenario=CustomScenario(),
        num_envs=1,
        device="cpu",
        continuous_actions=False,
        wrapper=None,
        max_steps=200,
        dict_spaces=False,
        n_agents=2,
)

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Model dimensions configurations
model = GCN(input_dim=3, hidden_dim=16, output_dim=9)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def create_graph_from_observations(observations):
    # Supponiamo che ogni agente abbia le osservazioni con la stessa dimensione
    node_features = [observations[f'agent{i}'] for i in range(len(observations))]
    
    # Stack dei tensori lungo la nuova dimensione (axis 0)
    node_features = torch.stack(node_features, dim=0)

    # Creare un grafo completo come esempio
    num_agents = node_features.size(0)
    edge_index = []
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            edge_index.append([i, j])
            edge_index.append([j, i])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    graph_data = Data(x=node_features, edge_index=edge_index)
    return graph_data


mock_observation = {'agent0': torch.tensor([[0.0081, 0.0411, 0.0000, 0.0000]]), 'agent1': torch.tensor([[-0.0021, -0.0179,  0.0000,  0.0000]])}
env.reset()
graph = create_graph_from_observations(mock_observation)

print(graph)

""" def train_step(graph_data, actions, rewards):
    model.train()
    optimizer.zero_grad()
    out = model(graph_data)
    loss = F.mse_loss(out, actions)  # Supponiamo di utilizzare MSE per il training
    loss.backward()
    optimizer.step()
    return loss.item() """

""" for episode in range(100):  # Numero di episodi di addestramento
    observations = env.reset()
    done = {agent: False for agent in env.agents}
    episode_loss = 0
    
    while not all(done.values()):
        graph_data = create_graph_from_observations(observations)
        actions = model(graph_data)
        
        actions_dict = {agent: actions[i].detach().numpy() for i, agent in enumerate(env.agents)}
        observations, rewards, done, _ = env.step(actions_dict)
        
        # Convert rewards to a tensor
        rewards_tensor = torch.tensor([rewards[agent] for agent in env.agents], dtype=torch.float)
        
        loss = train_step(graph_data, actions, rewards_tensor)
        episode_loss += loss
    
    print(f'Episode {episode}, Loss: {episode_loss}') """
