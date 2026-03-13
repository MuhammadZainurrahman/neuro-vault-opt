import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple

class PortfolioEnv:
    """
    Custom Environment for Portfolio Optimization.
    Simulates asset price dynamics and portfolio rebalancing.
    """
    def __init__(self, data: pd.DataFrame, initial_balance: float = 10000.0):
        self.data = data
        self.assets = data.columns.tolist()
        self.initial_balance = initial_balance
        self.reset()

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.weights = np.ones(len(self.assets)) / len(self.assets)
        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        # Returns current prices and technical indicators as a flattened vector
        return self.data.iloc[self.current_step].values

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """
        Executes a rebalancing action.
        Action: Target weights for each asset.
        """
        self.weights = self._softmax(action)
        
        # Calculate returns based on next step price change
        price_change = self.data.iloc[self.current_step + 1].values / self.data.iloc[self.current_step].values
        portfolio_return = np.dot(self.weights, price_change)
        
        reward = np.log(portfolio_return) # Log-return reward
        self.balance *= portfolio_return
        
        self.current_step += 1
        done = self.current_step >= len(self.data) - 2
        
        return self._get_observation(), reward, done

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

class PolicyNetwork(nn.Module):
    """
    Deep Neural Network for Actor-Critic Reinforcement Learning.
    """
    def __init__(self, input_dim: int, output_dim: int):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.actor = nn.Linear(64, output_dim)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.actor(x)), self.critic(x)

def train_agent(data: pd.DataFrame, epochs: int = 100):
    env = PortfolioEnv(data)
    model = PolicyNetwork(len(env.assets), len(env.assets))
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        state = torch.FloatTensor(env.reset())
        done = False
        epoch_reward = 0
        
        while not done:
            action_probs, value = model(state)
            next_state, reward, done = env.step(action_probs.detach().numpy())
            
            # Simplified PPO loss logic (for demonstration)
            # In a real scenario, use buffer and advantage estimation
            target_value = torch.tensor([reward])
            advantage = target_value - value
            
            loss = advantage**2 # Minimal Value Loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            state = torch.FloatTensor(next_state)
            epoch_reward += reward
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Reward = {epoch_reward:.4f}")

if __name__ == "__main__":
    # Sample synthetic data for testing
    dates = pd.date_range("2024-01-01", periods=100)
    data = pd.DataFrame(np.random.randn(100, 3).cumsum(axis=0) + 100, columns=['BTC', 'ETH', 'SOL'])
    train_agent(data)
