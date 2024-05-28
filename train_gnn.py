import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from vmas import make_env
from custom_scenario import CustomScenario
from torch_geometric.data import Data
import random

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
        self.conv3 = GCNConv(hidden_dim, hidden_dim, add_self_loops=False, bias=True)
        self.conv4 = GCNConv(hidden_dim, output_dim, add_self_loops=False, bias=True)
        self.apply(weights_init)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)
        x = torch.relu(x)
        x = self.conv4(x, edge_index)
        return x

def create_graph_from_observations(observations):
    node_features = [observations[f'agent{i}'] for i in range(len(observations))]
    node_features = torch.stack(node_features, dim=0).squeeze(dim=1)
    
    # Normalizza le caratteristiche dei nodi
    node_features = (node_features - node_features.mean(dim=0)) / (node_features.std(dim=0) + 1e-8)
    
    # Aggiungi un identificatore unico per ogni agente
    agent_ids = torch.arange(len(observations)).float().unsqueeze(1)
    node_features = torch.cat([node_features, agent_ids], dim=1)
    
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
    num_actions = 9  
    model = GCN(input_dim=5, hidden_dim=32, output_dim=num_actions)  # Aggiunto un neurone per l'ID agente
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    def train_step(graph_data, actions, rewards):
        model.train()
        optimizer.zero_grad()
        logits = model(graph_data)

        log_probs = F.log_softmax(logits, dim=1)
        selected_log_probs = log_probs[range(len(actions)), actions]

        loss = -torch.mean(selected_log_probs * rewards)
        
        l2_lambda = 0.001  # Ridotto per evitare overfitting
        l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
        loss += l2_lambda * l2_norm
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        return loss.item()

    epsilon = 0.1  
    epsilon_decay = 0.995  # Introduzione di epsilon decay
    min_epsilon = 0.01

    for episode in range(100):  
        observations = env.reset()
        episode_loss = 0
        
        for step in range(100):
            graph_data = create_graph_from_observations(observations)
            
            logits = model(graph_data)
            #print(f'Logits: {logits}')  

            if random.random() < epsilon:
                actions = torch.tensor([random.randint(0, num_actions - 1) for _ in range(len(env.agents))])
            else:
                actions = torch.argmax(logits, dim=1)

            actions_dict = {f'agent{i}': torch.tensor([actions[i].item()]) for i in range(len(env.agents))}
            observations, rewards, done, _ = env.step(actions_dict)
            #print(actions_dict) 
            rewards_tensor = torch.tensor([rewards[f'agent{i}'] for i in range(len(env.agents))], dtype=torch.float)
            
            rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
            
            loss = train_step(graph_data, actions, rewards_tensor)
            episode_loss += loss

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        average_loss = episode_loss / 100
        print(f'Episode {episode}, Loss: {average_loss}, Epsilon: {epsilon}')

    print("Training completed")
    torch.save(model.state_dict(), 'gcn_model.pth')
    print("Model saved successfully!")

if __name__ == "__main__":
    train_model()
