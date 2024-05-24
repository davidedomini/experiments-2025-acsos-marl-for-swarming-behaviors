from gnn import GCN
import torch
from vmas import make_env
from custom_scenario import CustomScenario
import random
from vmas.simulator.utils import save_video
import time
from gnn import create_graph_from_observations

# Crea un'istanza del modello
model = GCN(input_dim=2, hidden_dim=16, output_dim=9)
# Carica i pesi del modello
model.load_state_dict(torch.load('gcn_model.pth'))
model.eval()
print("Model loaded successfully!")

env = make_env(
    CustomScenario(),
    scenario_name = "test_gcn_vmas",
    num_envs=1,
    device="cpu",
    continuous_actions=False,
    dict_spaces=True,
    wrapper=None,
    seed=None,
    # Environment specific variables
    n_agents=2,
)
render=True
save_render=False
frame_list = []  # For creating a gif
init_time = time.time()
n_steps = 200
step = 0
total_reward = 0

observations = env.reset() #Dictionary that contains all the agents observations

for _ in range(n_steps):
    step += 1
    print(f"Step {step}")

    graph_data = create_graph_from_observations(observations)
        
    with torch.no_grad():
        logits = model(graph_data)
        actions = torch.argmax(logits, dim=1)
    
    actions_dict = {f'agent{i}': torch.tensor([actions[i].item()]) for i in range(len(env.agents))}
    
    observations, rewards, done, _ = env.step(actions_dict)

    total_reward += sum(rewards.values())

    if render:
        frame = env.render(
            mode="rgb_array",
            agent_index_focus=None,  # Can give the camera an agent index to focus on
            visualize_when_rgb=True,
        )
        if save_render:
            frame_list.append(frame)

total_time = time.time() - init_time
print(
    f"It took: {total_time}s for {n_steps} steps of {env.num_envs} parallel environments on device {env.device} "
    f"for test_gcn_vmas scenario."
)

if render and save_render:
    save_video("test_gcn_vmas", frame_list, fps=1 / env.scenario.world.dt)