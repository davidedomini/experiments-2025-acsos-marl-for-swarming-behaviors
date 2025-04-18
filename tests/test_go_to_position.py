from curses.ascii import SI
import sys
import os
from vmas import make_env
import torch

scenarios_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'scenarios'))
training_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'training'))
simulation_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'simulation'))

sys.path.insert(0, scenarios_dir)
sys.path.insert(1, training_dir)
sys.path.insert(2, simulation_dir)

from train_gcn_dqn import GCN
from go_to_position_scenario import GoToPositionScenario
from simulator import Simulator

if __name__ == "__main__":

    # 0 to 9
    models_seed = [i for i in range(10)]
    simulation_seed = 6967
    # 10 to 30
    agents = [i for i in range(10, 30)]

    for model_seed in models_seed:
        for agent in agents:
            env = make_env(
                GoToPositionScenario(),
                scenario_name="test_gcn_vmas",
                num_envs=1,
                device="cpu",
                continuous_actions=False,
                dict_spaces=True,
                wrapper=None,
                seed=simulation_seed,
                n_agents=agent,
                max_steps=50,
                random=True,
            )

            models_dir = "models/"

            model = GCN(input_dim=7, hidden_dim=32, output_dim=9)
            model.load_state_dict(torch.load(models_dir + f'experiment_GoTo-seed_{model_seed}.pth'))

            model.eval()
            print("Go to model loaded successfully!")

            simulator = Simulator(env, model, 8, 'go_to', simulation_seed,
                                  output_dir=f'test_stats/go_to/seed_{model_seed}/agents_{agent}')
            simulator.run_simulation()
