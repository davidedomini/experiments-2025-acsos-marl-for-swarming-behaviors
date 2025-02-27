# Multi-Agent Reinforcement Learning of Swarm Behaviors with Graph Neural Networks

This repository contains the implementation of a `Multi-Agent System` designed to train a swarm of autonomous agents and enable them to learn various swarming behaviors.

Training is accomplished through a custom implementation of the `Deep Q-Network` (DQN) algorithm and the design of a `Graph Neural Network` (GNN).

## Requirements 

All the requirements for run the project are listed in the file *requirements.txt*.

## Running the Simulations

To run the simulations, clone the repository and execute the desired test scripts found in the */tests/* directory.

## Swarming behaviours

The swarming behaviours learned by the agents are: **Go to position** (1) and **Obstacle Avoidance** (2). 

<p align="center">
  <table>
    <tr>
      <td style="text-align: center;">
        <img src="/media/go_to_position.gif" alt="Swarm Behavior 1" width="300"/>
        <p>(1) Go to Position</p>
      </td>
      <td style="text-align: center;">
        <img src="/media/obstacle_avoidance.gif" alt="Swarm Behavior 3" width="300"/>
        <p>(2) Obstacle Avoidance</p>
      </td>
    </tr>
  </table>
</p>

## Instruction for reproducing all the experiments

This repository contains a `docker-compose.yml` file that creates four services:
  * `prepare`: setups all the folders needed to save learning stats and models
  * `learning`: launches the training in all the different scenarios described in the paper
  * `test-goto-position`: uses the models saved during train to launch the scalability test in the GoTo Scenario
  * `test-obstacle-avoidance`: uses the models saved during train to launch the scalability test in the Obstacle Avoidance Scenario

### Running the experiments
* To run the complete process: `docker-compose up --build`
* Since learning may require a huge amount of time depending on which machine experiments are run to complete, we provided already trained models to run only the tests, in this case:
  * `docker compose run --no-deps test-goto-position` or
  * `docker compose run --no-deps test-obstacle-avoidance`
