# Deep-Q-Network-for-Maze-Solving

##Reinforcement learning coursework for MSc AI

Implemented a deep Q-network to solve randomly generated mazes. The implementation is divided in three files: random_environment.py, train_and_test.py, and agent.py (random_environment.py and train_and_test.py were given)

-random_environment.py creates random mazes - the agent always starts on the left of the maze and the goal is always on the right

-train_and_test.py gets an environment from random_environment.py, trains the agent for 10 minutes, and tests the agent's optimal policy on the maze

-agent.py consists of a double Q-network. It has the positions of the obstacles encoded in the reward function, and different levels of greedines and decay throughout the training period
