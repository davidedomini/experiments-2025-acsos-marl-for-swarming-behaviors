# Multi-Agent Reinforcement Learning of Swarm Behaviors with Graph Neural Networks

This repository contains the implementation of a `Multi-Agent System` designed to train a swarm of autonomous agents and enable them to learn various swarming behaviors.

Training is accomplished through a custom implementation of the `Deep Q-Network` (DQN) algorithm and the design of a `Graph Neural Network` (GNN).

## Requirements 

All the requirements for run the project are listed in the file *requirements.txt*.

## Running the Simulations

To run the simulations, clone the repository and execute the desired test scripts found in the */tests/* directory.

## Swarming behaviours

The swarming behaviours learned by the agents are: **Go to position** (1), **Flocking** (2) and **Obstacle Avoidance** (3). 

<p align="center">
  <table>
    <tr>
      <td style="text-align: center;">
        <img src="/media/go_to_position.gif" alt="Swarm Behavior 1" width="300"/>
        <p>(1) Go to Position</p>
      </td>
      <td style="text-align: center;">
        <img src="/media/flocking.gif" alt="Swarm Behavior 2" width="300"/>
        <p>(2) Flocking</p>
      </td>
      <td style="text-align: center;">
        <img src="/media/obstacle_avoidance.gif" alt="Swarm Behavior 3" width="300"/>
        <p>(3) Obstacle Avoidance</p>
      </td>
    </tr>
  </table>
</p>
