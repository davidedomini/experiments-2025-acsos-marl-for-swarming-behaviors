import torch
from train_gcn_dqn import create_graph_from_observations
import time

class Simulator:
    def __init__(self, env, model):
        self.env = env
        self.model = model

    def run_simulation(self):
        init_time = time.time()
        total_reward = 0

        observations = self.env.reset() 

        for step in range(self.env.max_steps):
            print(f"Step {step+1}")

            graph_data = create_graph_from_observations(observations)

            with torch.no_grad():
                logits = self.model(graph_data)

                print("Logits: ",logits)
                
                actions = torch.argmax(logits, dim=1)
            
            actions_dict = {f'agent{i}': torch.tensor([actions[i].item()]) for i in range(len(self.env.agents))}
            
            observations, rewards, _ , _ = self.env.step(actions_dict)

            total_reward += sum(rewards.values())

            frame = self.env.render(
                mode="rgb_array",
                agent_index_focus=None,
                visualize_when_rgb=True,
            )

            total_time = time.time() - init_time
            print(
                f"It took: {total_time}s for {self.env.max_steps} steps of {self.env.num_envs} parallel self.environments with {total_reward} total reward, on device {self.env.device} "
                f"for test_gcn_vmas scenario."
            )

