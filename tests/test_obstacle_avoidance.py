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
from obstacle_avoidance_scenario import ObstacleAvoidanceScenario
from simulator import Simulator

if __name__ == "__main__":
    # 0 to 9
    models_seed = [i for i in range(0, 10)]
    simulation_seed = 6967
    # 5 to 12
    agents = [i for i in range(10, 30)]

    for model_seed in models_seed:
        for agent in agents:

            env = make_env(
                ObstacleAvoidanceScenario(),
                scenario_name="test_gcn_vmas",
                num_envs=1,
                device="cpu",
                continuous_actions=False,
                dict_spaces=True,
                wrapper=None,
                seed=simulation_seed,
                n_agents=agent,
                max_steps= 100,
                random=True,
            )

            models_dir = "data/models/"

            model = GCN(input_dim=7, hidden_dim=32, output_dim=9)
            model.load_state_dict(torch.load(models_dir + f'experiment_ObstacleAvoidance-seed_{model_seed}.pth'))

            model.eval()
            print(f"Obstacle Avoidance model loaded successfully!, seed {model_seed}")

            simulator = Simulator(env, model, 8, 'obstacle_avoidance', simulation_seed, output_dir=f'data/test_stats/obstacle_avoidance/seed_{model_seed}/agents_{agent}', render=False)
            simulator.run_simulation()
