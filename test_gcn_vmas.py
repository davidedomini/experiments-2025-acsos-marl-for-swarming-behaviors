from train_gcn_dqn import GCN
import torch
from vmas import make_env
from custom_scenario import CustomScenario
from train_gcn_dqn import create_graph_from_observations
from vmas.simulator.utils import save_video
import time
from train_simple_nn import SimpleNN


model = GCN(input_dim=5, hidden_dim=32, output_dim=9)# Crea un'istanza del modello
model.load_state_dict(torch.load('gcn_model.pth'))# Carica i pesi del modello


""" model = SimpleNN() 
model.load_state_dict(torch.load('simple_nn_model.pth')) """

model.eval()
print("Model loaded successfully!")

env = make_env(
    CustomScenario(),
    scenario_name="test_gcn_vmas",
    num_envs=1,
    device="cpu",
    continuous_actions=False,
    dict_spaces=True,
    wrapper=None,
    seed=None,
    n_agents=5,
)

render = True
save_render = False
frame_list = [] 
init_time = time.time()
n_steps = 100
total_reward = 0

observations = env.reset() 

for step in range(n_steps):
    print(f"Step {step+1}")

    graph_data = create_graph_from_observations(observations)
    #graph_data = torch.cat([observations[f'agent{i}'] for i in range(len(env.agents))], dim=0)

    with torch.no_grad():
        logits = model(graph_data)

        #DEBUG: stampa i logits restituiti dal modello
        print("Logits: ",logits)
        
        actions = torch.argmax(logits, dim=1)
    
    actions_dict = {f'agent{i}': torch.tensor([actions[i].item()]) for i in range(len(env.agents))}
    
    observations, rewards, done, _ = env.step(actions_dict)

    total_reward += sum(rewards.values())

    if render:
        frame = env.render(
            mode="rgb_array",
            agent_index_focus=None,
            visualize_when_rgb=True,
        )
        if save_render:
            frame_list.append(frame)

total_time = time.time() - init_time
print(
    f"It took: {total_time}s for {n_steps} steps of {env.num_envs} parallel environments with {total_reward} total reward, on device {env.device} "
    f"for test_gcn_vmas scenario."
)

if render and save_render:
    save_video("test_gcn_vmas", frame_list, fps=1 / env.scenario.world.dt)
