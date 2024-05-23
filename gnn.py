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
model = GCN(input_dim=4, hidden_dim=16, output_dim=1)

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

import torch
import torch.nn.functional as F
import torch.optim as optim

# Supponiamo che il tuo modello sia chiamato 'model' e l'ottimizzatore 'optimizer'

def train_step(graph_data, actions, rewards_tensor):
    optimizer.zero_grad()
    
    # Aggiungi una loss arbitraria per testare il backpropagation
    test_loss = F.mse_loss(actions, torch.ones_like(actions))
    
    # Stampa la loss temporanea
    #print(f'Temporary Test Loss: {test_loss.item()}')
    
    test_loss.backward()
    
    # Debug: Stampa dei gradienti dopo il backward
    """ for name, param in model.named_parameters():
        if param.grad is not None:
            print(f'Gradient for {name}: {param.grad.norm()}') """
    
    optimizer.step()
    
    return test_loss.item()

for episode in range(100):  # Numero di episodi di addestramento
    observations = env.reset()
    episode_loss = 0

    graph_data = create_graph_from_observations(observations)
    
    actions = model(graph_data)

    # Debug: Stampa delle osservazioni
    print(f'Observations: {observations}')

    # Debug: Stampa delle azioni
    print(f'Actions: {actions}')


    # Debug: stampa delle azioni
    #print(f'Episode {episode}, Actions: {actions}')

    #actions_dict = {f'agent{i}': actions[i].detach().numpy() for i in range(len(env.agents))}
    actions_dict = {'agent0': torch.tensor([[1]]), 'agent1': torch.tensor([[1]])}
    observations, rewards, done, _ = env.step(actions_dict)
    
    rewards_tensor = torch.tensor([rewards[f'agent{i}'] for i in range(len(env.agents))], dtype=torch.float)

    loss = train_step(graph_data, actions, rewards_tensor)
    episode_loss += loss
    
    # Debug: stampa della loss
    print(f'Episode {episode}, Loss: {episode_loss}')
    
    """ # Debug: verifica degli aggiornamenti del modello
    with torch.no_grad():
        for name, param in model.named_parameters():
            print(f'Parameter {name} after update: {param.data}') """
