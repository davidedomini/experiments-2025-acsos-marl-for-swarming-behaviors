## Execute the GCN experiment

**Train the GCN**: execute *train_gcn.py*

**Simulate one run in VMAS**: execute *test_gcn_vmas.py*

**Change the environment configuration** (eg: Reward Structure): modify *custom_scenario.py*

**Debug:**

- **train_gcn**: Enable logs for print observations, rewards, graph features, loss and logits.

- **test_gcn_vmas**: Enable logs for print logits selected by the model.

## Actual configuration of the GCN training

- **Graph Convolutional Network** with:

    - 4 layer
    - 32 hidden dimensions
    - 5 input (posX, posY, velX, velY, agentID)
    - 9 output (from 0 to 8, all the possible movement actions of the agents)
	


- **Reward structure**:
	- reward basing on the distance from the goal.
	- check the actual distance of the agent from the goal, if it's bigger than the previous step, the reward decrease otherwise it becomes bigger

- **Training step**:
    - 100 episodes with 100 step
    - **Adam** as **Optimizer** with a learning rate of 0.001
	- **log likelihood** for calculate the loss
	- **L2 regularization** for avoiding overfitting from 0.001 to 0.1
	- **clip of the gradient** for avoid gradient vanishing/explosion from 1.0 to 5.0
	- **epsilon-greedy** for explore with epsilon=0.1 and epsilon_decay=0.995
	- **reward normalization** by global parameters (mean and variance) calculated at every step, with the reward already collected
