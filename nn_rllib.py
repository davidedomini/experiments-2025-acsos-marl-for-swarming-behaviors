import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.agents.ppo import PPOTrainer
from ray import tune
import ray
from vmas import make_env, Wrapper
from custom_scenario import CustomScenario
from ray.rllib.env import MultiAgentEnv
from ray.tune import register_env
from typing import Dict
import numpy as np


class CustomModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CustomModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

torch, nn = try_import_torch()

class RLLibCustomModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        input_dim = int(np.product(obs_space.shape))
        output_dim = num_outputs

        self.custom_model = CustomModel(input_dim, output_dim)
        
    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        print(obs)
        logits = self.custom_model(obs)
        return logits, state

    @override(ModelV2)
    def value_function(self):
        # Optional: define a value function
        return torch.zeros(1)

# Register the custom model
ModelCatalog.register_custom_model("rllib_custom_model", RLLibCustomModel)

def env_creator(config: Dict):
    env = make_env(
        scenario=CustomScenario(),
        num_envs=1,
        device="cpu",
        continuous_actions=False,
        wrapper=Wrapper.RLLIB,
        max_steps=200,
        dict_spaces=True,
        n_agents=3,
    )
    obs = env.env.reset()
    return env

if not ray.is_initialized():
    ray.init()

register_env("custom_vmas_env", lambda config: env_creator(config))

tune.run(
    PPOTrainer,
    config={
        "env": "custom_vmas_env",
        "model": {
            "custom_model": "rllib_custom_model",
        },
        "framework": "torch",
    },
    stop={"episode_reward_mean": 200},
)