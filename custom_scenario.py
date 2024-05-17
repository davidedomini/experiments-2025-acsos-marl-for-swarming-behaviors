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
        # Make world
        world = World(batch_dim, device)
        
        # Add agents
        agent1 = Agent(
            name="agent1",
            collide=True,
            color=Color.GREEN,
            render_action=True,
            #action_script=self.action_script_creator(),
        )
        world.add_agent(agent1)
       
        agent2 = Agent(
                name=f"agent2",
                collide=True,
                render_action=True,
            )

        world.add_agent(agent2)

        return world

    def reset_world_at(self, env_index: int = None):
        center_pos = torch.zeros(
            (1, 2) if env_index is not None else (self.world.batch_dim, 2),
            device=self.world.device,
            dtype=torch.float32,
        )

        # Set the agent's position at the center
        for agent in self.world.agents:
            agent.set_pos(
                center_pos,
                batch_index=env_index,
            )

    def action_script_creator(self):
        def action_script(agent, world):
            print(torch.tensor([[0.1, 0.1]]))
            agent.action.u = torch.tensor([[0.1, 0.1]])

        return action_script

    def reward(self, agent: Agent):
        # Simple reward structure (could be more sophisticated if needed)
        return torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.float32)

    def observation(self, agent: Agent):
        # Return the agent's position and velocity
        return torch.cat(
            [
                agent.state.pos,
                agent.state.vel,
            ],
            dim=-1,
        )

    def done(self):
        # Define the termination condition (if any, currently none)
        return torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.bool)

    def info(self, agent: Agent):
        # Provide additional information if needed
        return {}

    
if __name__ == "__main__":
    render_interactively(CustomScenario(), control_two_agents=False)