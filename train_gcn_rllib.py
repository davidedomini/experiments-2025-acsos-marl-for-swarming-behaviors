import torch
import torch.nn.functional as F
from gym.spaces import Discrete
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch
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
from test_gcn_rllib import use_vmas_env
import os

RLLIB_NUM_GPUS = int(os.environ.get("RLLIB_NUM_GPUS", "0"))
num_gpus = 0.001 if RLLIB_NUM_GPUS > 0 else 0  # Driver GPU
num_workers = 6
vmas_device = "cpu"  # or cuda
num_gpus_per_worker = (
    (RLLIB_NUM_GPUS - num_gpus) / (num_workers + 1) if vmas_device == "cuda" else 0
)

class GNNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, add_self_loops=False, bias=True)
        self.conv2 = GATConv(hidden_dim, hidden_dim, add_self_loops=False, bias=True)
        self.conv3 = GATConv(hidden_dim, hidden_dim, add_self_loops=False, bias=True)
        self.lin1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)
        x = torch.relu(x)
        x = self.lin1(x)
        x = torch.relu(x)
        x = self.lin2(x)
        return x

def build_graph(observations):
    # Input observations: Tensor of shape [num_agents, batch_size, num_features]
    num_agents, batch_size, num_features = observations.shape

    #print("BATCH_SIZE: ", batch_size, " NUM_AGENTS: ", num_agents, " NUM_FEATURES: ", num_features)

    # Reshape to create node features
    # Node features shape: [batch_size * num_agents, num_features]
    node_features = observations.permute(1, 0, 2).reshape(batch_size * num_agents, num_features)

    #print("NODE_FEATURES SHAPE: ", node_features.shape)

    # Create edge index
    edge_index = []
    for b in range(batch_size):
        for i in range(num_agents):
            for j in range(num_agents):
                if i != j:  # Ensure no self-loops
                    edge_index.append([b * num_agents + i, b * num_agents + j])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    graph_data = Data(x=node_features, edge_index=edge_index)
    return graph_data

class CustomGNNModel(TorchModelV2, torch.nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        torch.nn.Module.__init__(self)

        input_dim = 6  # Number of features per agent (6)
        hidden_dim = model_config.get("custom_model_config", {}).get("hidden_dim", 32)
        output_dim = 9  # Should match the number of actions (9)

        self.gnn = GNNModel(input_dim, hidden_dim, output_dim)

    def forward(self, input_dict, state, seq_lens):
        agent_states = torch.stack(input_dict["obs"])

        #print("AGENT_STATES SHAPE: ", agent_states.shape)

        num_agents, batch_size, _ = agent_states.shape

        graph_data = build_graph(agent_states)
        logits = self.gnn(graph_data)

        #print("LOGITS BEFORE: ", logits.shape)

        # Reshape logits to [batch_size, num_agents, output_dim]
        logits = logits.view(batch_size, num_agents, -1)

        #print("LOGITS MID: ", logits.shape)

        # Flatten logits to [batch_size, num_agents * output_dim]
        logits = logits.view(batch_size, -1)

        #print("LOGITS AFTER: ", logits.shape)

        return logits, state

    def value_function(self):
        return torch.zeros(1)

ModelCatalog.register_custom_model("custom_gnn_model", CustomGNNModel)

config = {
    "seed": 0,
    "framework": "torch",
    "env": "custom_vmas_env",
    "kl_coeff": 0.01,
    "kl_target": 0.01,
    "lambda": 0.9,
    "clip_param": 0.2,
    "vf_loss_coeff": 1,
    "vf_clip_param": float("inf"),
    "entropy_coeff": 0,
    "lr": 5e-5,
    "gamma": 0.99,
    "use_gae": True,
    "use_critic": True,
    "batch_mode": "truncate_episodes",
    "evaluation_interval": 5,
    "evaluation_duration": 1,
    "evaluation_num_workers": 1,
    "evaluation_parallel_to_training": True,
    "evaluation_config": {
        "num_envs_per_worker": 64,
        "env_config": {
            "num_envs": 64,
        }
    },
    "env_config": {
        "num_agents": 5,
    },
    "model": {
        "custom_model": "custom_gnn_model",
        "custom_model_config": {
            "hidden_dim": 32,
        },
    },
    "framework": "torch",
    "num_workers": 4,
    
}
""" "train_batch_size": 60000,
    "rollout_fragment_length": 125,
    "sgd_minibatch_size": 4096,
    "num_sgd_iter": 40,
    "num_envs_per_worker": 64, """
""" config = {
    "seed": 0,
    "framework": "torch",
    "env": "custom_vmas_env",
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
    "num_envs_per_worker": 64,
    "ignore_worker_failures": True,
    "lr": 5e-5,
    "gamma": 0.99,
    "use_gae": True,
    "use_critic": True,
    "batch_mode": "truncate_episodes",
    "evaluation_interval": 5,
    "evaluation_duration": 1,
    "evaluation_num_workers": 1,
    "evaluation_parallel_to_training": True,
    "model": {
        "custom_model": "custom_gnn_model",
        "custom_model_config": {
            "hidden_dim": 32,
        },
    },
    "env_config": {
        "num_agents": 2,
    },
    "evaluation_config": {
        "num_envs_per_worker": 64,
        "env_config": {
            "num_envs": 64,
        }
    }
} """

def env_creator(config: Dict):
    env = make_env(
        scenario=CustomScenario(),
        num_envs=1,
        device="cpu",
        continuous_actions=False,
        wrapper=Wrapper.RLLIB,
        max_steps=200,
        dict_spaces=False,
        n_agents=2,
    )
    return env

if not ray.is_initialized():
    ray.init()

register_env("custom_vmas_env", lambda config: env_creator(config))

def train():
    """ res = tune.run(
        PPOTrainer,
        stop={"training_iteration": 100},
        checkpoint_freq=1,
        keep_checkpoints_num=2,
        checkpoint_at_end=True,
        checkpoint_score_attr="episode_reward_mean",
        config=config,
        metric="episode_reward_mean",  # Specifica la metrica
        mode="max"  # Specifica la modalità di ottimizzazione
    ) """

    trainer = PPOTrainer(config=config)
    #trainer.restore(res.best_checkpoint)

    #DIFFERENT POS 2 (DIFFERENT REWARDS): /home/filippo/ray_results/PPO_2024-06-25_12-38-45/PPO_custom_vmas_env_192f7_00000_0_2024-06-25_12-38-45/checkpoint_000099
    #GO TO 2: /home/filippo/ray_results/PPO_2024-06-25_11-12-21/PPO_custom_vmas_env_07218_00000_0_2024-06-25_11-12-21/checkpoint_000100
    #FLOCKING: /home/filippo/ray_results/PPO_2024-06-25_13-51-06/PPO_custom_vmas_env_34938_00000_0_2024-06-25_13-51-06/checkpoint_000100
    #DIFFERENT POS 2 (SAME REWARDS): /home/filippo/ray_results/PPO_2024-06-25_15-03-56/PPO_custom_vmas_env_61732_00000_0_2024-06-25_15-03-56/checkpoint_000100

    trainer.restore("/home/filippo/ray_results/PPO_2024-06-25_15-03-56/PPO_custom_vmas_env_61732_00000_0_2024-06-25_15-03-56/checkpoint_000100")

    return trainer

if __name__ == "__main__":
    trainer = train()
    
    use_vmas_env(
        render=True,
        save_render=False,
        trainer=trainer,
    )
