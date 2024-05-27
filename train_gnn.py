import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from vmas import make_env
from custom_scenario import CustomScenario
from torch_geometric.data import Data

# Funzione per l'inizializzazione dei pesi
def weights_init(m):
    if isinstance(m, GCNConv):
        torch.nn.init.kaiming_uniform_(m.lin.weight, nonlinearity='relu')
        if m.lin.bias is not None:
            torch.nn.init.zeros_(m.lin.bias)

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
        self.conv1 = GCNConv(input_dim, hidden_dim, add_self_loops=False, bias=True)
        self.conv2 = GCNConv(hidden_dim, hidden_dim, add_self_loops=False, bias=True)
        self.conv3 = GCNConv(hidden_dim, output_dim, add_self_loops=False, bias=True)
        self.apply(weights_init)  # Applica l'inizializzazione dei pesi

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
    
    # Normalizza le caratteristiche dei nodi
    node_features = (node_features - node_features.mean(dim=0)) / (node_features.std(dim=0) + 1e-8)
    
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    def train_step(graph_data, actions, rewards):
        model.train()
        optimizer.zero_grad()
        out = model(graph_data)
        loss = F.cross_entropy(out, actions)
        
        # Aggiungi regularizzazione L2 per evitare overfitting
        l2_lambda = 0.01
        l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
        loss = loss + l2_lambda * l2_norm
        
        loss.backward()

        # Clipping dei gradienti
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Controlla i gradienti
        grad_norms = {name: param.grad.norm().item() if param.grad is not None else 0.0 for name, param in model.named_parameters()}
        #print(f'Gradient norms: {grad_norms}')

        optimizer.step()
        return loss.item()

    for episode in range(100):  # Numero di episodi di addestramento
        observations = env.reset()  # Reset dell'ambiente all'inizio di ogni episodio
        episode_loss = 0
        
        for step in range(100):  # Numero di passi per episodio
            graph_data = create_graph_from_observations(observations)
            
            logits = model(graph_data)
            #print(logits)
            actions = torch.argmax(logits, dim=1)  # Seleziona l'azione con il valore più alto

            actions_dict = {f'agent{i}': torch.tensor([actions[i].item()]) for i in range(len(env.agents))}
            observations, rewards, done, _ = env.step(actions_dict)
            print(observations)
            #print(actions_dict)
            
            rewards_tensor = torch.tensor([rewards[f'agent{i}'] for i in range(len(env.agents))], dtype=torch.float)
            print(rewards_tensor)
            
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
