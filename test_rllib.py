import os
from typing import Dict, Optional

import numpy as np
import ray
import wandb
from ray import tune
from ray.rllib import BaseEnv, Policy, RolloutWorker
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.algorithms.callbacks import DefaultCallbacks, MultiCallbacks
from ray.rllib.evaluation import Episode, MultiAgentEpisode
from ray.rllib.utils.typing import PolicyID
from ray.tune import register_env
from ray.tune.integration.wandb import WandbLoggerCallback
from vmas import make_env, Wrapper
from custom_scenario import CustomScenario

from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

def env_creator():
    env = make_env(
        scenario=CustomScenario(),
        num_envs=96,
        device="cpu",
        continuous_actions=True,
        wrapper=Wrapper.RLLIB,
        max_steps=200,
        # Scenario specific variables
        **{
            "n_agents": 1,
        },
    )
    return env

if not ray.is_initialized():
    ray.init()
    print("Ray init!")
register_env("custom_scenario", lambda config: env_creator())

model = (
    PPOConfig()
    .training(
        gamma = 0.95,
        lr = 0.0005,
        kl_coeff = 0.2,
        train_batch_size = 1,
        sgd_minibatch_size= 1,
        num_sgd_iter= 5
    )
    .resources(num_gpus = 0)
    .environment(env = "custom_scenario")
    .build()
)

def train():
    model.train()
    


if __name__ == "__main__":
    train()