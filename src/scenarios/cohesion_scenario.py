import torch
from vmas.simulator.core import Agent, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color
import numpy as np

class CohesionScenario(BaseScenario):

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):    
        self.pos_shaping_factor = kwargs.get("pos_shaping_factor", 10.0)
        self.dist_shaping_factor = kwargs.get("dist_shaping_factor", 10.0)
        self.agent_radius = kwargs.get("agent_radius", 0.1)
        self.n_agents = kwargs.get("n_agents", 1)

        self.min_distance_between_entities = self.agent_radius * 2 + 0.05
        self.world_semidim = 1

        self.agent_collision_reward = -1
        self.desired_distance = 0.15
        self.min_collision_distance = 0.005
        self.collective_reward = 0

        self.sigma = 0.15

        world = World(batch_dim, device)
        
        for i in range(self.n_agents):
            agent = Agent(
                name=f"agent{i}",
                collide=True,
                color=Color.GREEN,
                render_action=True,
            )
            agent.pos_rew = torch.zeros(batch_dim, device=device)
            agent.collision_rew = agent.pos_rew.clone()
            world.add_agent(agent)

        self.pos_rew = torch.zeros(batch_dim, device=device)
        self.final_rew = self.pos_rew.clone()

        return world


    def reset_world_at(self, env_index: int = None):

        all__agents_positions = torch.tensor([
            [-1.0, -1.0], 
            [0.0, -1.0],  
            [0.0, 1.0], 
            [0.0, 0.0],   
            [1.0, 1.0],
            [1.0, -1.0], 
            [-1.0, 1.0], 
            [1.0, 0.0],  
            [-1.0, 0.0],
        ], device='cpu', dtype=torch.float32)

        # Set the agents positions
        for i, agent in enumerate(self.world.agents):

            agent.set_pos(
                all__agents_positions[i],
                batch_index=env_index,
            )

    def reward(self, agent):
        distances = self.computeDistancesFromAgents(agent)

        min_distance = torch.min(distances)
        max_distance = torch.max(distances)

        #print(agent.name, " min: ", min_distance, " max: ", max_distance)

        return self.collision_factor(min_distance) + self.cohesion_factor(min_distance, max_distance)

    def computeDistancesFromAgents(self, agent: Agent):
        return torch.cat([self.world.get_distance(agent, other_agent) for other_agent in self.world.agents if agent.name != other_agent.name])
    
    def collision_factor(self, min_distance):
        return 0 if min_distance > self.sigma else np.exp(-(min_distance/self.sigma))
    
    def cohesion_factor(self, min_distance, max_distance):
        return 0 if min_distance < self.sigma else -(max_distance-self.sigma)

    def observation(self, agent: Agent):
        return torch.cat(
            [
                agent.state.pos,
                agent.state.vel
            ],
            dim=-1,
        )

    def done(self):
        return torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.bool)

    def info(self, agent: Agent):
        return {
            "pos_rew": agent.pos_rew,
            "final_rew": self.final_rew
        }
