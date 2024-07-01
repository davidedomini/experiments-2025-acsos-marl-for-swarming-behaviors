import random
import time
import sys
import os

scenarios_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'scenarios'))
sys.path.insert(0, scenarios_dir)

import torch
from vmas import make_env, Wrapper

from vmas import make_env
from vmas.simulator.core import Agent
from vmas.simulator.utils import save_video
from custom_scenario import CustomScenario
from ray.rllib.algorithms.ppo import PPO
import numpy as np
from vmas.simulator.environment.rllib import VectorEnvWrapper
from ray.rllib.agents.ppo import PPOTrainer

def use_vmas_env(
    render: bool = False,
    save_render: bool = False,
    n_steps: int = 200,
    scenario_name: str = "custom_scenario",
    visualize_render: bool = True,
    trainer: PPO = None,
    env_config: dict = None,
):
    assert not (save_render and not render), "To save the video you have to render it"

    env = make_env(
        scenario=CustomScenario(),
        num_envs=1,
        device="cpu",
        continuous_actions=False,
        wrapper=Wrapper.RLLIB,
        max_steps=200,
        dict_spaces=False,
        n_agents=2,
    ).env
    frame_list = []  # For creating a gif
    init_time = time.time()
    step = 0

    obs_batch = env.reset() #Dictionary that contains all the agents observations
    
    for _ in range(n_steps):
        step += 1
        print(f"Step {step}")
        
        obs_batch = torch.cat(obs_batch, dim=1)
            
        actions, state_out, actions_info = trainer.get_policy().compute_actions(obs_batch=obs_batch)

        actions_list = [torch.tensor(a) for a in actions]
        
        obs_batch, rews, dones, info = env.step(actions_list)

        if render:
            frame = env.render(
                mode="rgb_array",
                agent_index_focus=None,  # Can give the camera an agent index to focus on
                visualize_when_rgb=visualize_render,
            )
            if save_render:
                frame_list.append(frame)

    total_time = time.time() - init_time
    print(
        f"It took: {total_time}s for {n_steps} steps of {env.num_envs} parallel environments on device {env.device} "
        f"for custom scenario."
    )

    if render and save_render:
        save_video(scenario_name, frame_list, fps=1 / env.scenario.world.dt)