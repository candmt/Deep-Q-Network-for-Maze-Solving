# Deep_Q_Network_for_Maze_Solving

## Reinforcement learning coursework for MSc AI at Imperial College London, academic year 2020-2021

Implemented a deep Q-network to solve randomly generated mazes. The implementation is divided in three files: random_environment.py, train_and_test.py, and agent.py (random_environment.py and train_and_test.py were given)

* random_environment.py creates random mazes - the agent always starts on the left of the maze and the goal is always on the right

* train_and_test.py gets an environment from random_environment.py, trains the agent for 10 minutes, and tests the agent's optimal policy on the maze

* Features implemented in the agent.py file:
    - Double Q-network 
    - Epsilon-greedy policy. Different levels of epsilon decay throughout the training
    - Episode length variations across the training time
    - Experience replay buffer
    - Positions of the obstacles encoded in the reward function by penalizing the agent every time it hits a wall 
