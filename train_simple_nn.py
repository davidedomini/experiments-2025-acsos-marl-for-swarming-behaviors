import torch
import torch.nn as nn
import torch.nn.functional as F
from vmas import make_env
from custom_scenario import CustomScenario
import random

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 9)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_model():
    num_actions = 9
    model = SimpleNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    def train_step(observations, actions, rewards):
        model.train()
        optimizer.zero_grad()
        logits = model(observations)
        
        log_probs = F.log_softmax(logits, dim=1)
        selected_log_probs = log_probs[range(len(actions)), actions]
        
        loss = -torch.mean(selected_log_probs * rewards)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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

    obs_min = torch.tensor([float('inf')] * 4)
    obs_max = torch.tensor([float('-inf')] * 4)

    for episode in range(100):
        observations = env.reset()
        episode_loss = 0
        total_episode_reward = torch.zeros(env.n_agents)
        all_rewards = []
        all_observations = []

        for step in range(100):
            obs_tensor = torch.cat([observations[f'agent{i}'] for i in range(len(env.agents))], dim=0)

            obs_min = torch.min(obs_min, obs_tensor.min(dim=0)[0])
            obs_max = torch.max(obs_max, obs_tensor.max(dim=0)[0])

            normalized_obs = (obs_tensor - obs_min) / (obs_max - obs_min + 1e-8)
            normalized_obs = 2 * (normalized_obs - 0.5)

            logits = model(normalized_obs)

            if random.random() < epsilon:
                actions = torch.tensor([random.randint(0, num_actions - 1) for _ in range(len(env.agents))])
            else:
                actions = torch.argmax(logits, dim=1)

            actions_dict = {f'agent{i}': torch.tensor([actions[i].item()]) for i in range(len(env.agents))}
            observations, rewards, done, _ = env.step(actions_dict)

            rewards_tensor = torch.tensor([rewards[f'agent{i}'] for i in range(len(env.agents))], dtype=torch.float)
            all_rewards.append(rewards_tensor)
            all_observations.append(normalized_obs)

            total_episode_reward += rewards_tensor
            
            loss = train_step(normalized_obs, actions, rewards_tensor)
            episode_loss += loss

            #print(f"observations: {normalized_obs}, actions: {actions_dict}, Rewards: {rewards_tensor}")

        all_rewards = torch.stack(all_rewards)
        max_reward, _ = torch.max(all_rewards, dim=0)
        min_reward, _ = torch.min(all_rewards, dim=0)
        
        normalized_rewards = (all_rewards - min_reward) / (max_reward - min_reward + 1e-8)
        normalized_rewards = 2 * (normalized_rewards - 0.5)

        for step in range(100):
            loss = train_step(all_observations[step], actions, normalized_rewards[step])
            episode_loss += loss

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        average_loss = episode_loss / 100
        print(f'Episode {episode}, Loss: {average_loss}, Reward: {total_episode_reward}, Epsilon: {epsilon}')

    print("Training completed")
    torch.save(model.state_dict(), 'simple_nn_model.pth')
    print("Model saved successfully!")

if __name__ == "__main__":
    train_model()
