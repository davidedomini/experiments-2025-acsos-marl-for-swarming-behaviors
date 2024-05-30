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
    
    # Aggiunge un identificatore unico per ogni agente
    agent_ids = torch.arange(len(observations)).float().unsqueeze(1)
    
    node_features = torch.cat([node_features, agent_ids], dim=1)

    # DEBUG: osservazioni prese come input e node_features create
    #print(observations)
    #print(node_features)
    
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    global_reward_mean = 0.0
    global_reward_var = 0.0
    global_reward_count = 0

    def update_global_reward_stats(rewards):
        nonlocal global_reward_mean, global_reward_var, global_reward_count
        global_reward_count += rewards.size(0)
        delta = rewards - global_reward_mean
        global_reward_mean += delta.sum() / global_reward_count
        delta2 = rewards - global_reward_mean
        global_reward_var += delta.mul(delta2).sum()
    
        # DEBUG: aggiornamento incrementale della media e varianza dei reward per la normalizzazione
        #print(f"Updated stats - Mean: {global_reward_mean}, Var: {global_reward_var}, Count: {global_reward_count}")


    def get_normalized_rewards(rewards):
        std = (global_reward_var / global_reward_count).sqrt()
        std = std + 1e-8  # Aggiunta di epsilon per evitare divisioni per zero
        normalized_rewards = (rewards - global_reward_mean) / std
        
        # DEBUG: reward prima e dopo la normalizzazione
        #print(f"Rewards: {rewards}, Normalized rewards: {normalized_rewards}")
        
        return normalized_rewards


    def train_step(graph_data, actions, rewards):
        model.train()  # Imposta il modello in modalità di addestramento
        optimizer.zero_grad()  # Azzera i gradienti dei parametri del modello
        logits = model(graph_data)  # Calcola i logit del modello dati i dati del grafo
        
        log_probs = F.log_softmax(logits, dim=1)  # Calcola i log delle probabilità softmax
        selected_log_probs = log_probs[range(len(actions)), actions]  # Seleziona i log delle probabilità corrispondenti alle azioni effettuate
        
        loss = -torch.mean(selected_log_probs * rewards)  # Calcola la loss negativa della log likelihood pesata per le ricompense
        
        l2_lambda = 0.01  # Coefficiente di regolarizzazione L2
        l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())  # Calcola la norma L2 dei pesi del modello
        loss += l2_lambda * l2_norm  # Aggiunge il termine di regolarizzazione L2 alla loss
        
        loss.backward()  # Esegue la retropropagazione del gradiente
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Applica la clip del gradiente per evitare problemi di esplosione del gradiente
        optimizer.step()  # Esegue un passaggio di ottimizzazione utilizzando l'ottimizzatore

        # DEBUG: loss dello step, distribuzione di probabilità per l'azione di ogni agente, rewards
        #print(f"Loss: {loss.item()}, Selected log probs: {selected_log_probs}, Rewards: {rewards}")

        return loss.item() 

    epsilon = 0.3  
    epsilon_decay = 0.995 
    min_epsilon = 0.01

    for episode in range(100):  
        observations = env.reset()
        episode_loss = 0
        total_episode_reward = torch.zeros(env.n_agents)  
        
        for step in range(100):
            graph_data = create_graph_from_observations(observations)
            
            logits = model(graph_data)

            # DEBUG: logits restituiti in output dalla GCN
            #print(f'Logits: {logits}')  

            if random.random() < epsilon:
                actions = torch.tensor([random.randint(0, num_actions - 1) for _ in range(len(env.agents))])
            else:
                actions = torch.argmax(logits, dim=1)

            actions_dict = {f'agent{i}': torch.tensor([actions[i].item()]) for i in range(len(env.agents))}
            observations, rewards, done, _ = env.step(actions_dict)

            rewards_tensor = torch.tensor([rewards[f'agent{i}'] for i in range(len(env.agents))], dtype=torch.float)
            
            update_global_reward_stats(rewards_tensor)
            normalized_rewards = get_normalized_rewards(rewards_tensor)
            
            total_episode_reward += normalized_rewards
            
            loss = train_step(graph_data, actions, normalized_rewards)
            episode_loss += loss

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        average_loss = episode_loss / 100
        print(f'Episode {episode}, Loss: {average_loss}, Reward: {total_episode_reward}, Epsilon: {epsilon}')

    print("Training completed")
    torch.save(model.state_dict(), 'gcn_model.pth')
    print("Model saved successfully!")

if __name__ == "__main__":
    train_model()
