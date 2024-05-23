from typing import Callable, Dict

import torch
from torch import Tensor
from vmas import render_interactively
from vmas.simulator.core import Agent, Landmark, Sphere, World, Entity
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import Color, X, Y, ScenarioUtils

class CustomScenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):    
        self.pos_shaping_factor = kwargs.get("pos_shaping_factor", 1)
        self.agent_radius = kwargs.get("agent_radius", 0.1)
        self.n_agents = kwargs.get("n_agents", 1)

        self.min_distance_between_entities = self.agent_radius * 2 + 0.05
        self.world_semidim = 1

        # Make world
        world = World(batch_dim, device)

        # Add goals
        goal = Landmark(
            name=f"goal",
            collide=False,
            color=Color.BLACK,
        )
        world.add_landmark(goal)

        for i in range(self.n_agents):
            # Add agents
            agent = Agent(
                name=f"agent{i}",
                collide=True,
                color=Color.GREEN,
                render_action=True,
            )
            agent.pos_rew = torch.zeros(batch_dim, device=device)
            agent.goal = goal
            world.add_agent(agent)

        self.pos_rew = torch.zeros(batch_dim, device=device)
        self.final_rew = self.pos_rew.clone()

        return world

    def reset_world_at(self, env_index: int = None):
        center_pos = torch.zeros(
            (1, 2) if env_index is not None else (self.world.batch_dim, 2),
            device=self.world.device,
            dtype=torch.float32,
        )

        num_agents = len(self.world.agents)

        perturbation_scale = 0.1  

        perturbations = (torch.rand((num_agents, 2), device=self.world.device, dtype=torch.float32) - 0.5) * perturbation_scale

        # Set the agents' positions near the center
        for i, agent in enumerate(self.world.agents):
            
            perturbed_pos = center_pos + perturbations[i]
            agent.set_pos(
                perturbed_pos,
                batch_index=env_index,
            )
            if env_index is None:
                agent.pos_shaping = (
                    torch.linalg.vector_norm(
                        agent.state.pos - agent.goal.state.pos,
                        dim=1,
                    )
                    * self.pos_shaping_factor
                )
            else:
                agent.pos_shaping[env_index] = (
                    torch.linalg.vector_norm(
                        agent.state.pos[env_index] - agent.goal.state.pos[env_index]
                    )
                    * self.pos_shaping_factor
                )

    def reward(self, agent: Agent):
        return self.agent_reward(agent)
    
    def agent_reward(self, agent: Agent):
        agent.distance_to_goal = torch.linalg.vector_norm(
            agent.state.pos - agent.goal.state.pos,
            dim=-1,
        )
        agent.on_goal = agent.distance_to_goal < agent.goal.shape.radius

        pos_shaping = agent.distance_to_goal * self.pos_shaping_factor
        agent.pos_rew = agent.pos_shaping - pos_shaping
        agent.pos_shaping = pos_shaping
        return agent.pos_rew

    def observation(self, agent: Agent):
        # Return the agent's position and velocity
        return torch.cat(
            [
                agent.state.pos
            ],
            dim=-1,
        )

    def done(self):
        # Define the termination condition (if any, currently none)
        return torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.bool)

    def info(self, agent: Agent):
        return {
            "pos_rew": agent.pos_rew,
            "final_rew": self.final_rew
        }

    
if __name__ == "__main__":
    render_interactively(CustomScenario(), control_two_agents=False)