import torch
import torch.nn as nn
import os
import numpy as np
import torch.optim as optim
from torch.distributions import Categorical
from typing import Tuple, List
import torch.nn.functional as F
import warnings
warnings.simplefilter(action='ignore')
class PPOMemory:
    """
    A memory buffer for storing experiences during training with the Proximal Policy Optimization (PPO) algorithm.

    Parameters
    ----------
    batch_size : int
        The batch size used when generating training batches.

    Attributes
    ----------
    states : list
        A list to store state observations.
    probs : list
        A list to store the log probabilities of the actions taken.
    vals : list
        A list to store value estimates from the critic network.
    actions : list
        A list to store actions taken.
    rewards : list
        A list to store rewards received after taking actions.
    dones : list
        A list to store done flags indicating episode termination.
    batch_size : int
        The batch size used for generating training batches.
    """
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Generates batches of experiences for training.

        Returns
        -------
        state_arr : np.ndarray
            An array of state observations.
        action_arr : np.ndarray
            An array of actions taken.
        prob_arr : np.ndarray
            An array of log probabilities of the actions taken.
        val_arr : np.ndarray
            An array of value estimates from the critic.
        reward_arr : np.ndarray
            An array of rewards received.
        done_arr : np.ndarray
            An array of done flags indicating episode termination.
        batches : List[np.ndarray]
            A list of batches, where each batch is an array of indices.
        """
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        return (
            np.stack(self.states),
            np.array(self.actions),
            np.array(self.probs),
            np.array(self.vals),
            np.array(self.rewards),
            np.array(self.dones),
            batches,
        )

    def store_memory(self, state, action, probs, vals, reward, done):
        """
        Stores a single transition in memory.

        Parameters
        ----------
        state : torch.Tensor
            The current state observation.
        action : int
            The action taken.
        probs : float
            The log probability of the action taken.
        vals : float
            The value estimate from the critic network.
        reward : float
            The reward received after taking the action.
        done : bool
            Whether the episode has terminated after this step.
        """
        self.states.append(state.cpu().numpy())
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        """
        Clears all stored experiences from memory.
        """
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

class Encoder(nn.Module):
    """
    Convolutional encoder network that processes observations and outputs embeddings.

    Parameters
    ----------
    input_channels : int, default=3
        Number of channels in the input observations.
    embedding_dim : int, default=128
        Dimension of the output embedding.
    input_height : int, default=21
        Height of the input observations.
    input_width : int, default=21
        Width of the input observations.

    Attributes
    ----------
    conv1 : nn.Conv2d
        First convolutional layer.
    conv2 : nn.Conv2d
        Second convolutional layer.
    fc : nn.Linear
        Fully connected layer that outputs the embedding.
    flattened_size : int
        The size of the flattened feature map after the convolutional layers.
    device : torch.device
        The device on which the model is allocated.
    """
    def __init__(self, input_channels=3, embedding_dim=128, input_height=21, input_width=21):
        super(Encoder, self).__init__()
        self.conv = nn.Conv2d(input_channels, 2, kernel_size=3, stride=1, padding=1)

        # Compute the size of the flattened output after conv layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, input_height, input_width)
            x = torch.relu(self.conv(dummy_input))
            self.flattened_size = x.view(1, -1).size(1)

        # Initialize the fully connected layer with the computed size
        self.fc = nn.Linear(self.flattened_size, embedding_dim)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        """
        Forward pass through the encoder network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_channels, input_height, input_width).

        Returns
        -------
        x : torch.Tensor
            Output embedding tensor of shape (batch_size, embedding_dim).
        """
        x = torch.relu(self.conv(x))
        x = x.reshape(x.size(0), -1)  # Use reshape instead of view
        x = torch.relu(self.fc(x))
        return x

class ActorNetwork(nn.Module):
    """
    Actor network that outputs a policy distribution over actions.

    Parameters
    ----------
    input_channel : int
        Number of channels for the CNN encoder.
    embedding_dim : int
        Dimension of the input embedding from the encoder.
    n_actions : int
        Number of possible actions.
    alpha : float
        Learning rate for the optimizer.
    fc1_dims : int, default=256
        Number of units in the first fully connected layer.
    fc2_dims : int, default=256
        Number of units in the second fully connected layer.
    chkpt_dir : str, default='../experiments/ppo'
        Directory to save and load model checkpoints.

    Attributes
    ----------
    encoder : nn.Module
        Shared encoder network.
    actor : nn.Sequential
        Sequential model defining the actor network.
    device : torch.device
        The device on which the model is allocated.
    checkpoint_file : str
        Path to the model checkpoint file.
    """
    def __init__(self, input_channel, embedding_dim, n_actions, alpha, fc1_dims=32, fc2_dims=16, chkpt_dir='../experiments/ppo'):
        super(ActorNetwork, self).__init__()
        self.checkpoint_file = os.path.join('..', chkpt_dir, 'actor_torch_ppo')
        self.encoder = Encoder(input_channels=input_channel, embedding_dim=embedding_dim)
        self.actor = nn.Sequential(
            self.encoder,
            nn.Linear(embedding_dim, fc1_dims),
            nn.LeakyReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.LeakyReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1)
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        """
        Forward pass through the actor network to get action probabilities.

        Parameters
        ----------
        state : torch.Tensor
            Input tensor representing the state.

        Returns
        -------
        dist : torch.distributions.Categorical
            Categorical distribution over actions.
        """
        dist = self.actor(state)
        dist = Categorical(dist)
        return dist

    def save_checkpoint(self):
        """
        Saves the model parameters to the checkpoint file.
        """
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        """
        Loads the model parameters from the checkpoint file.
        """
        self.load_state_dict(torch.load(self.checkpoint_file))

class CriticNetwork(nn.Module):
    """
    Critic network that estimates the value function.

    Parameters
    ----------
    input_channel : int
        Number of channels for the CNN encoder.
    embedding_dim : int
        Dimension of the input embedding from the encoder.
    alpha : float
        Learning rate for the optimizer.
    fc1_dims : int, default=256
        Number of units in the first fully connected layer.
    fc2_dims : int, default=256
        Number of units in the second fully connected layer.
    chkpt_dir : str, default='../experiments/ppo'
        Directory to save and load model checkpoints.

    Attributes
    ----------
    encoder : nn.Module
        Shared encoder network.
    critic : nn.Sequential
        Sequential model defining the critic network.
    device : torch.device
        The device on which the model is allocated.
    checkpoint_file : str
        Path to the model checkpoint file.
    """
    def __init__(self, input_channel, embedding_dim, alpha, fc1_dims=32, fc2_dims=16, chkpt_dir='../experiments/ppo'):
        super(CriticNetwork, self).__init__()
        self.checkpoint_file = os.path.join('..', chkpt_dir, 'critic_torch_ppo')
        self.encoder = Encoder(input_channels=input_channel, embedding_dim=embedding_dim)
        
        self.critic = nn.Sequential(
            self.encoder,
            nn.Linear(embedding_dim, fc1_dims),
            nn.LeakyReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.LeakyReLU(),
            nn.Linear(fc2_dims, 1)
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        """
        Forward pass through the critic network to get the value estimate.

        Parameters
        ----------
        state : torch.Tensor
            Input tensor representing the state.

        Returns
        -------
        value : torch.Tensor
            Estimated value of the input state.
        """
        value = self.critic(state)
        return value

    def save_checkpoint(self):   
        """
        Saves the model parameters to the checkpoint file.
        """
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        """
        Loads the model parameters from the checkpoint file.
        """
        self.load_state_dict(torch.load(self.checkpoint_file))
class Agent:
    """
    Agent class implementing the PPO algorithm for training and action selection.

    Parameters
    ----------
    n_actions : int
        Number of possible actions.
    input_channels : int, default=3
        Number of channels in the input observations.
    input_height : int, default=21
        Height of the input observations.
    input_width : int, default=21
        Width of the input observations.
    embedding_dim : int, default=128
        Dimension of the embedding output by the encoder.
    gamma : float, default=0.99
        Discount factor for future rewards.
    alpha : float, default=0.0003
        Learning rate for the optimizer.
    gae_lambda : float, default=0.95
        Lambda parameter for Generalized Advantage Estimation.
    policy_clip : float, default=0.2
        Clipping parameter for PPO policy loss.
    batch_size : int, default=64
        Batch size for training.
    n_epochs : int, default=10
        Number of epochs to train for each update.

    Attributes
    ----------
    gamma : float
        Discount factor for future rewards.
    policy_clip : float
        Clipping parameter for PPO policy loss.
    n_epochs : int
        Number of epochs to train for each update.
    gae_lambda : float
        Lambda parameter for Generalized Advantage Estimation.
    batch_size : int
        Batch size for training.
    shared_encoder : nn.Module
        Shared encoder network for processing observations.
    actor : ActorNetwork
        Actor network for policy estimation.
    critic : CriticNetwork
        Critic network for value estimation.
    memory : PPOMemory
        Memory buffer for storing experiences.
    optimizer : torch.optim.Optimizer
        Optimizer for updating network parameters.
    """
    def __init__(self, n_actions, input_channels=3, input_height=21, input_width=21, embedding_dim=128, gamma=0.99,
                 alpha=0.0003, gae_lambda=0.95, policy_clip=0.2, batch_size=64, n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.batch_size = batch_size


        self.actor = ActorNetwork(input_channel=input_channels,
                                  embedding_dim=embedding_dim, n_actions=n_actions, alpha=alpha)
        self.critic = CriticNetwork(input_channel=input_channels,
                                    embedding_dim=embedding_dim, alpha=alpha)
        self.memory = PPOMemory(self.batch_size)

        # Create the optimizer without duplicate parameters
        self.optimizer = optim.Adam([
            {'params': self.actor.actor.parameters()},
            {'params': self.critic.critic.parameters()},
        ], lr=alpha)

    def remember(self, state, action, probs, vals, reward, done):
        """
        Stores an experience in memory.

        Parameters
        ----------
        state : torch.Tensor
            The current state observation.
        action : int
            The action taken.
        probs : float
            The log probability of the action taken.
        vals : float
            The value estimate from the critic network.
        reward : float
            The reward received after taking the action.
        done : bool
            Whether the episode has terminated after this step.
        """
        self.memory.store_memory(state.cpu(), action, probs, vals, reward, done)

    def save_models(self):
        """
        Saves the actor and critic models to their respective checkpoint files.
        """
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        """
        Loads the actor and critic models from their respective checkpoint files.
        """
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        """
        Selects an action based on the current observation.

        Parameters
        ----------
        observation : torch.Tensor
            The current observation/state.

        Returns
        -------
        action : int
            The action selected.
        probs : float
            The log probability of the action taken.
        value : float
            The value estimate from the critic network.
        """
        if len(observation.shape) == 3:
            observation = observation.unsqueeze(0)
        state = observation.to(self.actor.device)
        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()
        probs = dist.log_prob(action).squeeze().item()
        action = action.squeeze().item()
        value = value.squeeze().item()
        return action, probs, value

    def learn(self):
        """
        Performs a learning step, updating the actor and critic networks using the experiences stored in memory.
        """
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = \
                self.memory.generate_batches()

            # Convert arrays to tensors with correct data types
            values = torch.tensor(vals_arr, dtype=torch.float32, device=self.actor.device)
            rewards = torch.tensor(reward_arr, dtype=torch.float32, device=self.actor.device)
            dones = torch.tensor(dones_arr, dtype=torch.float32, device=self.actor.device)
            advantage = torch.zeros(len(rewards), dtype=torch.float32, device=self.actor.device)

            # Convert scalars to tensors
            gamma = torch.tensor(self.gamma, dtype=torch.float32, device=self.actor.device)
            gae_lambda = torch.tensor(self.gae_lambda, dtype=torch.float32, device=self.actor.device)

            # Compute advantage
            for t in range(len(rewards) - 1):
                discount = torch.tensor(1.0, dtype=torch.float32, device=self.actor.device)
                a_t = torch.tensor(0.0, dtype=torch.float32, device=self.actor.device)
                for k in range(t, len(rewards) - 1):
                    delta = rewards[k] + gamma * values[k + 1] * (1 - dones[k]) - values[k]
                    a_t += discount * delta
                    discount *= gamma * gae_lambda
                advantage[t] = a_t

            advantage = advantage.detach()
            values = values.detach()

            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float32, device=self.actor.device)
                actions = torch.tensor(action_arr[batch], dtype=torch.long, device=self.actor.device)
                old_probs = torch.tensor(old_prob_arr[batch], dtype=torch.float32, device=self.actor.device)
                returns = advantage[batch] + values[batch]

                dist = self.actor(states)
                critic_value = self.critic(states).squeeze()

                new_probs = dist.log_prob(actions)
                prob_ratio = torch.exp(new_probs - old_probs)
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                critic_loss = F.mse_loss(critic_value, returns)

                total_loss = actor_loss + 0.5 * critic_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

        self.memory.clear_memory()
