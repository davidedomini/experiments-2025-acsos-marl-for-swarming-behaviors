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
        dict_spaces=True,
        n_agents=2,
)

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    import torch.nn.functional as F

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return F.softmax(x, dim=1)


# Model dimensions configurations
model = GCN(input_dim=2, hidden_dim=16, output_dim=1)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def create_graph_from_observations(observations):
    # Supponiamo che ogni agente abbia le osservazioni con la stessa dimensione
    node_features = [observations[f'agent{i}'] for i in range(len(observations))]
    
    # Stack dei tensori lungo la nuova dimensione (axis 0)
    node_features = torch.stack(node_features, dim=0).squeeze(dim=1)

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


#mock_observation = {'agent0': torch.tensor([[0.0081, 0.0411, 0.0000, 0.0000]]), 'agent1': torch.tensor([[-0.0021, -0.0179,  0.0000,  0.0000]])}
#env.reset()
#graph_observations = create_graph_from_observations(mock_observation)

# Supponiamo che il tuo modello sia chiamato 'model' e l'ottimizzatore 'optimizer'


# Supponiamo che il tuo modello sia chiamato 'model' e l'ottimizzatore 'optimizer'

def train_step(graph_data, actions, rewards):
    model.train()
    optimizer.zero_grad()
    out = model(graph_data)
    loss = F.mse_loss(out, actions)  # Supponiamo di utilizzare MSE per il training
    loss.backward()
    optimizer.step()
    return loss.item()

observations = env.reset()

for episode in range(100):  # Numero di episodi di addestramento
    
    episode_loss = 0

    graph_data = create_graph_from_observations(observations)
    
    actions = model(graph_data)

    print(f'Observations: {observations}')

    print(f'Actions: {actions}')

    actions_dict = {f'agent{i}': actions[i].detach().numpy() for i in range(len(env.agents))}
    #actions_dict = {'agent0': torch.tensor([[1]]), 'agent1': torch.tensor([[2]])}
    observations, rewards, done, _ = env.step(actions_dict)
    
    rewards_tensor = torch.tensor([rewards[f'agent{i}'] for i in range(len(env.agents))], dtype=torch.float)

    loss = train_step(graph_data, actions, rewards_tensor)
    episode_loss += loss

