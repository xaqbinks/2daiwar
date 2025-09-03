# src/ai_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """
    Deep Q-Network model for the agent.
    """
    def __init__(self, n_observations, n_actions):
        """
        Initializes the neural network layers.

        :param n_observations: The size of the state space.
        :param n_actions: The number of possible actions.
        """
        super(DQN, self).__init__()
        # A simple feed-forward network
        self.layer1 = nn.Linear(n_observations, 128) # Input layer
        self.layer2 = nn.Linear(128, 128)             # Hidden layer
        self.layer3 = nn.Linear(128, n_actions)      # Output layer

    def forward(self, x):
        """
        Defines the forward pass of the network.

        :param x: The input state tensor.
        :return: The Q-values for each action.
        """
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


from collections import namedtuple
import random

# A named tuple 'Transition' represents a single transition in our environment.
# It maps a (state, action) pair to its (next_state, reward) result,
# with a 'done' flag indicating if the episode has ended.
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory(object):
    """
    A cyclic buffer of bounded size that holds the transitions observed recently.
    It allows for efficient sampling of random transitions for training.
    """

    def __init__(self, capacity):
        """
        Initializes the replay memory.
        :param capacity: The maximum number of transitions to store.
        """
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Selects a random batch of transitions for training.
        :param batch_size: The number of transitions to sample.
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


import math
import torch.optim as optim

class DQNAgent:
    """
    The agent that encapsulates the DQN model and the learning process.
    """
    def __init__(self, n_observations, n_actions):
        self.n_observations = n_observations
        self.n_actions = n_actions

        # --- Hyperparameters ---
        self.batch_size = 128
        self.gamma = 0.99       # Discount factor
        self.eps_start = 0.9    # Starting value of epsilon
        self.eps_end = 0.05     # Final value of epsilon
        self.eps_decay = 1000   # Controls the rate of exponential decay of epsilon
        self.tau = 0.005        # The update rate of the target network
        self.lr = 1e-4          # Learning rate of the AdamW optimizer

        # --- Setup ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create the policy and target networks
        self.policy_net = DQN(n_observations, n_actions).to(self.device)
        self.target_net = DQN(n_observations, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict()) # Clone the weights

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.memory = ReplayMemory(10000)

        self.steps_done = 0

    def select_action(self, state):
        """
        Selects an action using an epsilon-greedy policy.
        """
        sample = random.random()
        # Calculate the current epsilon threshold
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1

        if sample > eps_threshold:
            # Exploit: choose the best action from the policy network
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            # Explore: choose a random action
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)

    def learn(self):
        """
        Performs one step of the optimization process.
        """
        if len(self.memory) < self.batch_size:
            return # Don't learn until we have enough memory

        transitions = self.memory.sample(self.batch_size)
        # Transposes the batch (see https://stackoverflow.com/a/19343/3343043 for details).
        # This converts a batch-array of Transitions to a Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a): the model computes Q(s_t), then we select the columns of actions taken.
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based on the "older" target_net.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]

        # Compute the expected Q values using the Bellman equation
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        # Soft update of the target network's weights
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target_net.load_state_dict(target_net_state_dict)
