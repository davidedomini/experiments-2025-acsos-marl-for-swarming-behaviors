from typing import Callable, Dict

import torch
import random
from torch import Tensor
from vmas import render_interactively
from vmas.simulator.core import Agent, Landmark, Sphere, World, Entity
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import Color, X, Y, ScenarioUtils

class CustomScenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):    
        self.pos_shaping_factor = kwargs.get("pos_shaping_factor", 10)
        self.agent_radius = kwargs.get("agent_radius", 0.1)
        self.n_agents = kwargs.get("n_agents", 1)

        self.min_distance_between_entities = self.agent_radius * 2 + 0.05
        self.world_semidim = 1

        self.collective_reward = 0
        self.collective_goal_reached_reward = 0

        world = World(batch_dim, device)

        goal = Landmark(
            name=f"goal",
            collide=False,
            color=Color.BLACK,
        )
        world.add_landmark(goal)
        
        for i in range(self.n_agents):
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
        # Definisce i limiti dell'area in cui posizionare gli agenti
        position_range = (-1, 1)
        
        num_agents = len(self.world.agents)
        num_landmarks = len(self.world.landmarks)

        """ # Genera posizioni casuali per i landmarks
        random_landmark_positions = (position_range[1] - position_range[0]) * torch.rand(
            (num_landmarks, 2), device=self.world.device, dtype=torch.float32
        ) + position_range[0]"""

        # Setta le posizioni dei landmarks alle posizioni casuali generate
        for i, landmark in enumerate(self.world.landmarks):
            landmark.set_pos(
                torch.tensor([-0.8, 0.8]),#random_landmark_positions[i],
                batch_index=env_index,
            ) 

        # Genera posizioni casuali all'interno dell'area definita
        random_agent_positions = (position_range[1] - position_range[0]) * torch.rand(
            (num_agents, 2), device=self.world.device, dtype=torch.float32
        ) + position_range[0]

        random_agent_positions = torch.tensor([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]])

        # Setta le posizioni degli agenti alle posizioni casuali generate
        for i, agent in enumerate(self.world.agents):
            agent.set_pos(
                random_agent_positions[i],
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

    def reward(self, agent):
        if agent == self.world.agents[0]:
            self.collective_goal_reached_reward = 0
            self.collective_reward = 0

            collective_goal_reached = True
            for a in self.world.agents:
                self.collective_reward += self.agent_reward(a)
                collective_goal_reached = collective_goal_reached and a.on_goal

            if collective_goal_reached:
                self.collective_goal_reached_reward = 100

        return self.collective_reward + self.collective_goal_reached_reward
        """ total_reward = 0
        for agent in self.world.agents:
            total_reward += self.agent_reward(agent)
        return total_reward / len(self.world.agents) """
    
    def agent_reward(self, agent: Agent):
        agent.distance_to_goal = torch.linalg.vector_norm(
            agent.state.pos - agent.goal.state.pos,
            dim=-1,
        )
        agent.on_goal = agent.distance_to_goal < agent.goal.shape.radius 

        pos_shaping = agent.distance_to_goal * self.pos_shaping_factor #Distanza attuale tra l'agente e il goal pesandola per shaping_factor
        agent.pos_rew = agent.pos_shaping - pos_shaping #Reward in base alla differenza tra la distanza precedente e quella attuale
        agent.pos_shaping = pos_shaping #Salva la distanza per la prossima iterazione

        reward = agent.pos_rew

        if agent.on_goal:
            reward = reward + 10

        return reward

    def observation(self, agent: Agent):
        return torch.cat(
            [
                agent.state.pos,
                agent.state.vel,
                #self.world.landmarks[0].state.pos
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

    
if __name__ == "__main__":
    render_interactively(CustomScenario(), control_two_agents=False)