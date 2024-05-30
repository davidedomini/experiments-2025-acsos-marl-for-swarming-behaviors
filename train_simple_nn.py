import torch
import torch.nn as nn
import torch.nn.functional as F
from vmas import make_env
from custom_scenario import CustomScenario
import random

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(4, 128)  # Layer di input con 4 input e 128 neuroni
        self.fc2 = nn.Linear(128, 64) # Layer nascosto con 128 input e 64 neuroni
        self.fc3 = nn.Linear(64, 9)   # Layer di output con 64 input e 9 output (azioni)

    def forward(self, x):
        x = F.relu(self.fc1(x))       # Funzione di attivazione ReLU dopo il primo layer
        x = F.relu(self.fc2(x))       # Funzione di attivazione ReLU dopo il secondo layer
        x = self.fc3(x)               # Output layer, non applicare ReLU qui
        return x

def train_model():
    num_actions = 9  
    model = SimpleNN() 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)  # Learning rate ridotto

    def train_step(observations, actions, rewards):
        model.train()
        optimizer.zero_grad()
        logits = model(observations)
        
        log_probs = F.log_softmax(logits, dim=1)
        selected_log_probs = log_probs[range(len(actions)), actions]
        
        loss = -torch.mean(selected_log_probs * rewards)
        loss.backward()

        # Clipping dei gradienti
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Monitoraggio dei gradienti
        for param in model.parameters():
            if param.grad is not None:
                if torch.any(torch.isnan(param.grad)):
                    raise ValueError("I gradienti contengono NaN")
                if torch.any(torch.isinf(param.grad)):
                    raise ValueError("I gradienti contengono Inf")

        optimizer.step()
        
        return loss.item() 

    epsilon = 0.3  
    epsilon_decay = 0.995 
    min_epsilon = 0.01

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

    for episode in range(100):  
        observations = env.reset()
        episode_loss = 0
        total_episode_reward = torch.zeros(env.n_agents)  
        
        for step in range(100):
            obs_tensor = torch.cat([observations[f'agent{i}'] for i in range(len(env.agents))], dim=0)
            logits = model(obs_tensor)

            if random.random() < epsilon:
                actions = torch.tensor([random.randint(0, num_actions - 1) for _ in range(len(env.agents))])
            else:
                actions = torch.argmax(logits, dim=1)

            actions_dict = {f'agent{i}': torch.tensor([actions[i].item()]) for i in range(len(env.agents))}
            observations, rewards, done, _ = env.step(actions_dict)

            rewards_tensor = torch.tensor([rewards[f'agent{i}'] for i in range(len(env.agents))], dtype=torch.float)
            rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-5)
            rewards_tensor = torch.clamp(rewards_tensor, -1.0, 1.0)
            total_episode_reward += rewards_tensor
            
            loss = train_step(obs_tensor, actions, rewards_tensor)
            episode_loss += loss

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        average_loss = episode_loss / 100
        print(f'Episode {episode}, Loss: {average_loss}, Reward: {total_episode_reward}, Epsilon: {epsilon}')

    print("Training completed")
    torch.save(model.state_dict(), 'simple_nn_model.pth')
    print("Model saved successfully!")

if __name__ == "__main__":
    train_model()
