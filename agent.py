import numpy as np
import collections
import torch
import random
from statistics import mean
from matplotlib import pyplot as plt
import time


class Agent:

    # Function to initialise the agent
    def __init__(self):
        # Set the episode length
        self.episode_length = 500
        # Reset the total number of steps which the agent has taken
        self.num_steps_taken = 0
        # The state variable stores the latest state of the agent in the environment
        self.state = None
        # The action variable stores the latest action which the agent has applied to the environment
        self.action = None
        # Create a ReplayBuffer object
        self.replay_buffer = ReplayBuffer()
        # Create a DQN object
        self.dqn = DQN()
        # Initialize the mini-batch size
        self.batch_size = 2000
        # Initialize number of steps taken before updating the target q network's weights to the q network's weights
        self.update_weights = 1000
        # Initialize values needed for e-greedy policy
        self.epsilon = 0.9
        self.epsilon_end = 0.02
        self.epsilon_decay = 0.9997
        # Count number of episodes
        self.episodes = 0
        # Average loss per episode for plotting
        self.total_losses = []
        # list of what episode we are on for plotting loss curve (doesn't start at 0 since we have to wait
        # until the buffer is full)
        self.episode = []
        # action rewards
        self.episode_rewards = 0
        # total reward per episode
        self.rewards = []
        # list of what episode we are in for rewards' plot (this one starts at 0)
        self.repisode = []
        # flag that checks if we have reached the goal
        self.done = False
        # flag that checks if the greedy policy is being tested
        self.greedy = False
        # check number of steps per episode
        self.steps_per_episode = 0
        # timer
        self.start_time = time.time()

    # Function to check whether the agent has reached the end of an episode
    def has_finished_episode(self):
        if self.num_steps_taken % self.episode_length == 0:
            self.episodes += 1
            # Parameters needed to draw the loss and rewards functions
            self.repisode.append(self.episodes)
            self.rewards.append(self.episode_rewards)
            self.episode_rewards = 0
            self.steps_per_episode = 0
            if len(self.dqn.episode_loss) > 0:
                self.total_losses.append(mean(self.dqn.episode_loss))
                self.episode.append(self.episodes)
                self.dqn.episode_loss = []
            return True
        else:
            return False

    # Function to get the next action
    def get_next_action(self, state):
        self.steps_per_episode += 1
        # E-greedy policy with fixed 0.9 epsilon
        if self.episodes < 8:
            discrete_action = self.epsilon_action(state, 0.9)
        # E-greedy policy with decaying epsilon throughout the episodes
        elif self.episodes < 12:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            discrete_action = self.epsilon_action(state, self.epsilon)
        elif self.episodes < 22:
            # Greedy policy for the first 70 steps
            if self.steps_per_episode <= 70:
                # Turn greedy flag off
                self.greedy = False
                discrete_action = self.greedy_action(state)
            else:
                # Turn greedy flag off
                self.greedy = False
                discrete_action = self.epsilon_action(state, 0.9)

        elif self.episodes < 30:
            self.episode_length = 570
            self.greedy = False
            #E-greedy policy fixed 0.85 epsilon
            discrete_action = self.epsilon_action(state, 0.85)
        else:
            # Train the agent with the greedy policy
            if self.episodes % 2 == 0:
                self.greedy = False
                self.episode_length = 200
                discrete_action = self.greedy_action(state)
                #print('greedy')
            # Test the greedy policy witohut training the network
            else:
                # Turn the greedy flag on
                self.greedy = True
                self.episode_length = 100
                discrete_action = self.greedy_action(state)
                #print('fully greedy')
        action = self._discrete_action_to_continuous(discrete_action)
        # Update the number of steps which the agent has taken
        self.num_steps_taken += 1
        # Store the state; this will be used later, when storing the transition
        self.state = state
        # Store the discrete action; this will be used later, when storing the transition
        self.action = discrete_action
        return action

    def reached_goal(self, distance_to_goal):
        # Implement a time thing when  the distance_to_goal condition changes to try to ensure it gets within 0.5
        if time.time() > self.start_time + 30:
            if self.steps_per_episode <= 100 and distance_to_goal <= 0.03 and self.greedy:
                # If it reached the goal, turn the flag on
                self.done = True
        if time.time() > self.start_time + 570:
            if self.steps_per_episode <= 100 and distance_to_goal <= 0.5 and self.greedy:
                self.done = True

    # Get the greedy action
    def greedy_action(self, state):
        current_state_tensor = torch.tensor(state)
        #Get the Q values from the q network
        current_state_value = self.dqn.q_network.forward(current_state_tensor)
        # Select the action with the highest value
        discrete_action = torch.argmax(current_state_value)
        return discrete_action

    # Get the e-greedy action
    def epsilon_action(self, state, epsilon):
        # Get a random decimal number from 0 to 1. If the number is grater than epsilon, choose the greedy action
        # otherwise choose a random action
        if np.random.rand(1) > epsilon:
            current_state_tensor = torch.tensor(state)
            current_state_value = self.dqn.q_network.forward(current_state_tensor)
            discrete_action = torch.argmax(current_state_value)
        else:
            discrete_action = random.randint(0, 2)
        return discrete_action


    # Function to set the next state and distance, which resulted from applying action self.action at state self.state
    def set_next_state_and_distance(self, next_state, distance_to_goal):
        # Convert the distance to a reward
        reward = 0.4*(2**(1/2) - distance_to_goal)**2
        # Penalize the agent for running into walls
        if next_state[0] == self.state[0] and next_state[1] == self.state[1]:
            reward = reward - 0.008
        # Reward the agent for moving right
        if self.state[0] < next_state[0]:
            reward = reward + 0.09
        self.episode_rewards += reward
        # Create the transition
        transition = (self.state, self.action, reward, next_state)
        # Add the transition to the replay buffer
        self.replay_buffer.add_transition(transition)
        # Check if the agent reached the goal
        if self.episodes >= 30:
            self.reached_goal(distance_to_goal)
        # If it hasn't reached the goal, and it's not testing the greedy policy, then train the network
        if not self.done and not self.greedy:
            # Check if mini_batch has enough transitions. If so, use the mini_batch to train the q network
            if len(self.replay_buffer.buffer) >= self.batch_size:
                mini_batch = self.replay_buffer.mini_batch(self.batch_size)
                self.dqn.train_q_network(mini_batch)
            # Update weights of target q network if needed
            if self.num_steps_taken % self.update_weights == 0:
                self.dqn.update_target_weights()

    # Function to get the greedy action for a particular state
    def get_greedy_action(self, state):
        current_state = state
        current_state_tensor = torch.tensor(current_state)
        current_state_value = self.dqn.q_network(current_state_tensor)
        discrete_action = torch.argmax(current_state_value)
        action = self._discrete_action_to_continuous(discrete_action)
        return action

    # Function to change a discrete action to continous
    def _discrete_action_to_continuous(self, discrete_action):
        if discrete_action == 0:
            continuous_action = np.array([0.02, 0], dtype=np.float32)
        elif discrete_action == 2:
            continuous_action = np.array([0, -0.02], dtype=np.float32)
        elif discrete_action == 1:
            continuous_action = np.array([0, 0.02], dtype=np.float32)
        return continuous_action

    # Function to draw the loss vs the number of episodes
    def draw_loss(self, iterations, losses):
        fig, ax = plt.subplots()
        ax.set(xlabel='Episodes', ylabel='Loss', title='Loss Curve for Agent')
        ax.plot(iterations, losses, color='blue')
        plt.yscale('log')
        plt.show()
        fig.savefig("loss curve with target netwrork.png")

    # Function to draw the rewards vs the number of episodes
    def draw_rewards(self, iterations, rewards):
        fig, ax = plt.subplots()
        ax.set(xlabel='Episodes', ylabel='Rewards', title='Reward Curve for Agent')
        ax.plot(iterations, rewards, color='blue')
        plt.yscale('log')
        plt.show()
        fig.savefig("loss curve with target netwrork.png")



# The Network class inherits the torch.nn.Module class, which represents a neural network.
class Network(torch.nn.Module):

    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, input_dimension, output_dimension):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        # Define the network layers. This  network has two hidden layers, each with 100 units.
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=100)
        self.layer_2 = torch.nn.Linear(in_features=100, out_features=100)
        self.output_layer = torch.nn.Linear(in_features=100, out_features=output_dimension)

    # Function which sends some input data through the network and returns the network's output.
    # A ReLU activation function is used for both hidden layers, but the output layer has no
    # activation function (it is just a linear layer).
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        output = self.output_layer(layer_2_output)
        return output


# The DQN class determines how to train the above neural network.
class DQN:

    # The class initialisation function.
    def __init__(self):
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimension=2, output_dimension=3)
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how
        # big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=0.002)
        # Create the target Q-network
        self.target_q_network = Network(input_dimension=2, output_dimension=3)
        self.target_optimiser = torch.optim.Adam(self.target_q_network.parameters(), lr=0.002)
        self.episode_loss = []

    # Function that is called whenever we want to train the Q-network. Each call to this function takes in a
    # mini-batch of transitios containing the data we use to update the Q-network.
    def train_q_network(self, mini_batch):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        loss = self._calculate_loss(mini_batch)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the
        # Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        self.episode_loss.append(loss.item())
        #return loss.item()

    # Function to calculate the loss for a particular transition.
    def _calculate_loss(self, mini_batch):
        states, actions, rewards, next_states = zip(*mini_batch)
        # Transform the elements of each transition into tensors
        state_tensor = torch.tensor(states, dtype=torch.float32)
        action_tensor = torch.tensor(actions, dtype=torch.int64)
        reward_tensor = torch.tensor(rewards, dtype=torch.float32)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32)
        # this is the network's prediction for the highest Q values of each state-action pair
        network_prediction = self.q_network.forward(state_tensor).gather(dim=1,index=action_tensor.unsqueeze(-1)).squeeze(-1)
        # Get action with max value for next state across the mini batch with the target q - network
        max_a_next_state = torch.argmax(self.target_q_network.forward(next_states_tensor).detach(), 1)
        # Double q network
        double_network_prediction = self.q_network.forward(next_states_tensor).gather(dim=1,index=max_a_next_state.unsqueeze(-1)).squeeze(-1)
        # Loss of network's label minus network's prediction
        loss = torch.nn.MSELoss()(reward_tensor+0.9*double_network_prediction, network_prediction)
        return loss

    # This function updates the target network's weights
    def update_target_weights(self):
        # Get the weights from the Q-netwrok
        network_weights = self.q_network.state_dict()
        # Set the target network's weights equal to the Q-network's weights
        self.target_q_network.load_state_dict(network_weights)


# Experience replay buffer, to allow for training of transitions in mini-batches
class ReplayBuffer:

    def __init__(self):
        self.buffer = collections.deque(maxlen=7000)

    # Add transitions to the buffer
    def add_transition(self, transition):
        self.buffer.append(transition)

    # Create a mini-batch of randomly selected transitions
    def mini_batch(self, batch_size):
        minibatch_indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        mini_batch = [self.buffer[idx] for idx in minibatch_indices]
        return mini_batch
