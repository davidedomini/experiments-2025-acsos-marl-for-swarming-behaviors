import torch
import math
from vmas.simulator.core import Agent, Landmark, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color

class ObstacleAvoidanceScenario(BaseScenario):

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.random = kwargs.get("random", False)
        self.pos_shaping_factor = kwargs.get("pos_shaping_factor", 10.0)
        self.dist_shaping_factor = kwargs.get("dist_shaping_factor", 10.0)
        self.agent_radius = kwargs.get("agent_radius", 0.1)
        self.n_agents = kwargs.get("n_agents", 1)

        self.min_distance_between_entities = self.agent_radius * 2 + 0.05
        self.world_semidim = 1

        self.n_obstacles = 1
        self.obstacle_collision_reward = -10
        self.collective_reward = 0
        self.agent_collision_reward = -1
        self.desired_distance = 0.15
        self.min_collision_distance_reward = 1
        self.min_collision_distance_count = 0.2

        world = World(batch_dim, device)

        goal = Landmark(
                name=f"goal",
                collide=False,
                color=Color.BLACK,
            )
        world.add_landmark(goal)

        for i in range(self.n_obstacles):

            obstacle = Landmark(
                name="obstacle",
                collide=True,
                color=Color.RED
            )

            world.add_landmark(obstacle)
        
        for i in range(self.n_agents):
            agent = Agent(
                name=f"agent{i}",
                collide=True,
                color=Color.GREEN,
                render_action=True,
            )
            agent.pos_rew = torch.zeros(batch_dim, device=device)
            agent.collision_rew = agent.pos_rew.clone()
            agent.goal = goal
            world.add_agent(agent)

        self.pos_rew = torch.zeros(batch_dim, device=device)
        self.final_rew = self.pos_rew.clone()

        return world
    
    def generate_grid(self, center: torch.Tensor, num_points: int, distance: float):
        """
        Generate a grid of positions around a center point.

        Parameters:
        center (torch.Tensor): Coordinates of the center point (x, y).
        num_points (int): The total number of points (agents) in the grid.
        distance (float): The distance between each position in the grid.

        Returns:
        torch.Tensor: A tensor containing the positions in the grid.
        """
        x_center, y_center = center

        num_cols = math.ceil(math.sqrt(num_points))
        num_rows = math.ceil(num_points / num_cols)

        grid = []
        for i in range(num_rows):
            for j in range(num_cols):
                x = x_center + (j - (num_cols - 1) / 2) * distance
                y = y_center + (i - (num_rows - 1) / 2) * distance
                grid.append([x, y])
                if len(grid) >= num_points:
                    break
            if len(grid) >= num_points:
                break
        
        return torch.tensor(grid)


    def reset_world_at(self, env_index: int = None):
        #self.world.obstacle_hits = 0

        self.world.landmarks[0].set_pos(torch.tensor([-0.8, 0.8]), batch_index=env_index)

        self.world.landmarks[1].set_pos(torch.tensor([-0.1, 0.1]), batch_index=env_index)
        delta = torch.normal(mean=torch.tensor([0.0, 0.0]), std=torch.tensor([0.1, 0.1])) if self.random else torch.tensor([0.0, 0.0])

        central_position = torch.tensor([0.6, -0.6]) + delta

        all__agents_positions = self.generate_grid(central_position, self.n_agents, self.desired_distance)

        for i, agent in enumerate(self.world.agents):

            agent.set_pos(
                all__agents_positions[i],
                batch_index=env_index,
            )

            agent.previous_distance_to_goal = (
                torch.linalg.vector_norm(
                    agent.state.pos - agent.goal.state.pos,
                    dim=1,
                )
                * self.pos_shaping_factor
            )

            agent.previous_distance_to_agents = (
                torch.stack(
                    [
                        torch.linalg.vector_norm(
                            agent.state.pos - a.state.pos, dim=-1
                        )
                        for a in self.world.agents
                        if a != agent
                    ],
                    dim=1,
                )
                - self.desired_distance
            ).pow(2).mean(-1) * self.dist_shaping_factor

    def reward(self, agent):
        return self.distance_to_goal_reward(agent) + 2.5 * self.obstacle_avoidance_reward(agent)
    
    def distance_to_goal_reward(self, agent: Agent):
        agent.distance_to_goal = torch.linalg.vector_norm(
            agent.state.pos - agent.goal.state.pos,
            dim=-1,
        )

        return -agent.distance_to_goal

    def obstacle_avoidance_reward(self, agent: Agent):

        for i in range (1, self.n_obstacles + 1):
            if self.world.get_distance(agent, self.world.landmarks[i]) <= self.min_collision_distance_reward:
                return -(self.min_collision_distance_reward - self.world.get_distance(agent, self.world.landmarks[i]))

        return 0

    def observation(self, agent: Agent):
        return torch.cat(
            [
                agent.state.pos,
                agent.state.vel,
                agent.goal.state.pos,
            ],
            dim=-1,
        )

    def average_distance_to_goal(self):
        return torch.mean(torch.stack([agent.distance_to_goal for agent in self.world.agents]))

    def average_distance_to_obstacles(self):
        return torch.mean(torch.stack([self.world.get_distance(agent, self.world.landmarks[1]) for agent in self.world.agents]))

    def obstacles_hits(self):
        hits = torch.stack([self.world.get_distance(agent, self.world.landmarks[1]) <= self.min_collision_distance_count for agent in self.world.agents])
        count_all = torch.sum(hits)
        return count_all
    def done(self):
        return torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.bool)

    def info(self, agent: Agent):
        return {
            "pos_rew": agent.pos_rew,
            "final_rew": self.final_rew
        }