#  Copyright (c) 2022-2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import os
from typing import Dict, Optional

import numpy as np
import ray
from ray import tune
from ray.rllib import BaseEnv, Policy, RolloutWorker
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.algorithms.callbacks import DefaultCallbacks, MultiCallbacks
from ray.rllib.evaluation import Episode, MultiAgentEpisode
from ray.rllib.utils.typing import PolicyID
from ray.tune import register_env
from vmas import make_env, Wrapper
from custom_scenario import CustomScenario
from test_custom_scenario import use_vmas_env

scenario_name = "custom_environment"

# Scenario specific variables.
# When modifying this also modify env_config and env_creator
n_agents = 1

# Common variables
continuous_actions = False
max_steps = 200
num_vectorized_envs = 1
num_workers = 5
vmas_device = "cpu"  # or cuda
env = None

RLLIB_NUM_GPUS = int(os.environ.get("RLLIB_NUM_GPUS", "0"))
num_gpus = 0.001 if RLLIB_NUM_GPUS > 0 else 0  # Driver GPU
num_gpus_per_worker = (
    (RLLIB_NUM_GPUS - num_gpus) / (num_workers + 1) if vmas_device == "cuda" else 0
)

config = {
            "seed": 0,
            "framework": "torch",
            "env": scenario_name,
            "kl_coeff": 0.01,
            "kl_target": 0.01,
            "lambda": 0.9,
            "clip_param": 0.2,
            "vf_loss_coeff": 1,
            "vf_clip_param": float("inf"),
            "entropy_coeff": 0,
            "train_batch_size": 60000,
            "rollout_fragment_length": 125,
            "sgd_minibatch_size": 4096,
            "num_sgd_iter": 40,
            "num_gpus": num_gpus,
            "num_workers": num_workers,
            "num_gpus_per_worker": num_gpus_per_worker,
            "num_envs_per_worker": num_vectorized_envs,
            "lr": 5e-5,
            "gamma": 0.99,
            "use_gae": True,
            "use_critic": True,
            "batch_mode": "truncate_episodes",
            "env_config": {
                "device": vmas_device,
                "num_envs": num_vectorized_envs,
                "scenario_name": scenario_name,
                "continuous_actions": continuous_actions,
                "max_steps": max_steps,
                # Scenario specific variables
                "scenario_config": {
                    "n_agents": n_agents,
                },
            },
            "evaluation_interval": 5,
            "evaluation_duration": 1,
            "evaluation_num_workers": 1,
            "evaluation_parallel_to_training": True,
            "evaluation_config": {
                "num_envs_per_worker": 1,
                "env_config": {
                    "num_envs": 1,
                }
            }
        }


def env_creator(config: Dict):
    env = make_env(
        scenario=CustomScenario(),
        num_envs=config["num_envs"],
        device=config["device"],
        continuous_actions=config["continuous_actions"],
        wrapper=Wrapper.RLLIB,
        max_steps=config["max_steps"],
        # Scenario specific variables
        **config["scenario_config"],
    )
    return env

if not ray.is_initialized():
    ray.init()
    print("Ray init!")

register_env(scenario_name, lambda config: env_creator(config))


def train():
    res = tune.run(
        PPOTrainer,
        stop={"training_iteration": 2},
        checkpoint_freq=1,
        keep_checkpoints_num=2,
        checkpoint_at_end=True,
        checkpoint_score_attr="episode_reward_mean",
        callbacks=[
        ],
        config=config,
        metric="episode_reward_mean",  # Specifica la metrica
        mode="max"  # Specifica la modalità di ottimizzazione
    )

    trainer = PPOTrainer(config=config)
    trainer.restore(res.best_checkpoint)
    #trainer.restore("/home/filippo/ray_results/PPO_2024-05-22_15-30-22/PPO_custom_environment_708a0_00000_0_2024-05-22_15-30-22/checkpoint_000002")

    return trainer


if __name__ == "__main__":
    trainer = train()
    
    use_vmas_env(
        scenario_name="custom_scenario",
        render=True,
        save_render=False,
        random_action=False,
        continuous_actions=False,
        trainer=trainer,
        env_config = config["env_config"]
    )
    
