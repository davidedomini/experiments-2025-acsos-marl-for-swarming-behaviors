from curses.ascii import SI
import sys
import os

scenarios_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'scenarios'))
training_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'training'))
sys.path.insert(0, scenarios_dir)
sys.path.insert(1, training_dir)

import torch
from vmas import make_env
from cohesion_scenario import CohesionScenario
from go_to_position_scenario import GoToPositionScenario
from flocking_scenario import FlockingScenario
from obstacle_avoidance_scenario import ObstacleAvoidanceScenario
from train_gcn_dqn import create_graph_from_observations, GCN
from vmas.simulator.utils import save_video
import time

models_dir = "../models/"

model = GCN(input_dim=5, hidden_dim=32, output_dim=9)# Crea un'istanza del modello
model.load_state_dict(torch.load(models_dir + 'cohesion_collision.pth'))# Carica i pesi del modello

model.eval()
print("Model loaded successfully!")

class Simulator:
    def __init__(self, env, model):
        self.env = env
        self.model = model

    def run_simulation(self):
        init_time = time.time()
        total_reward = 0

        observations = env.reset() 

        for step in range(self.env.max_steps):
            print(f"Step {step+1}")

            graph_data = create_graph_from_observations(observations)

            with torch.no_grad():
                logits = self.model(graph_data)

                print("Logits: ",logits)
                
                actions = torch.argmax(logits, dim=1)
            
            actions_dict = {f'agent{i}': torch.tensor([actions[i].item()]) for i in range(len(env.agents))}
            
            observations, rewards, _ , _ = env.step(actions_dict)

            total_reward += sum(rewards.values())

            frame = env.render(
                mode="rgb_array",
                agent_index_focus=None,
                visualize_when_rgb=True,
            )

            total_time = time.time() - init_time
            print(
                f"It took: {total_time}s for {self.env.max_steps} steps of {env.num_envs} parallel environments with {total_reward} total reward, on device {env.device} "
                f"for test_gcn_vmas scenario."
            )



env = make_env(
    CohesionScenario(),
    scenario_name="test_gcn_vmas",
    num_envs=1,
    device="cpu",
    continuous_actions=False,
    dict_spaces=True,
    wrapper=None,
    seed=None,
    n_agents=9,
    max_steps= 100
)

if __name__ == "__main__":
    simulator = Simulator(env, model)
    simulator.run_simulation()

