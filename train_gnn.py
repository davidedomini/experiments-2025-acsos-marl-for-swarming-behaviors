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
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)
        return x


def create_graph_from_observations(observations):
    node_features = [observations[f'agent{i}'] for i in range(len(observations))]
    node_features = torch.stack(node_features, dim=0).squeeze(dim=1)
    num_agents = node_features.size(0)
    edge_index = []
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            edge_index.append([i, j])
            edge_index.append([j, i])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    graph_data = Data(x=node_features, edge_index=edge_index)
    return graph_data

def train_model():
    # Modello con output dimensione pari al numero di azioni possibili
    num_actions = 9  
    model = GCN(input_dim=4, hidden_dim=16, output_dim=num_actions)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    def train_step(graph_data, actions, rewards):
        model.train()
        optimizer.zero_grad()
        out = model(graph_data)
        loss = F.cross_entropy(out, actions)
        loss.backward()
        optimizer.step()
        return loss.item()

    for episode in range(100):  # Numero di episodi di addestramento
        observations = env.reset()  # Reset dell'ambiente all'inizio di ogni episodio
        episode_loss = 0
        
        for step in range(100):  # Numero di passi per episodio
            graph_data = create_graph_from_observations(observations)
            
            logits = model(graph_data)
            actions = torch.argmax(logits, dim=1)  # Seleziona l'azione con il valore più alto

            actions_dict = {f'agent{i}': torch.tensor([actions[i].item()]) for i in range(len(env.agents))}
            observations, rewards, done, _ = env.step(actions_dict)
            
            rewards_tensor = torch.tensor([rewards[f'agent{i}'] for i in range(len(env.agents))], dtype=torch.float)
            
            # Normalizza i rewards se necessario
            rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
            
            loss = train_step(graph_data, actions, rewards_tensor)
            episode_loss += loss

        # Per ogni episodio, calcola la perdita media
        average_loss = episode_loss / 100
        print(f'Episode {episode}, Loss: {average_loss}')

    print("Training completed")

    # Salva il modello addestrato
    torch.save(model.state_dict(), 'gcn_model.pth')
    print("Model saved successfully!")

if __name__ == "__main__":
    train_model()
