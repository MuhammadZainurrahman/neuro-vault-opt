# Neuro-Vault-Opt: Neural Portfolio Optimization Framework

[![Research](https://img.shields.io/badge/Research-ML-blue.svg)](Abstract.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An advanced research-driven framework for **Autonomous Portfolio Allocation** using **Proximal Policy Optimization (PPO)**. This repository implements a deep reinforcement learning agent designed to navigate the high-dimensional manifolds of volatile asset markets.

## 🔬 Core Methodology
- **Deep Reinforcement Learning**: Implements an Actor-Critic architecture for real-time weight rebalancing.
- **Custom Environment**: A high-fidelity financial environment simulating transaction costs, slippage, and multi-asset price dynamics.
- **Mathematical Optimization**: Uses log-return reward functions and stochastic gradient updates for policy convergence.

## 🛠 Project Structure
- `src/engine.py`: Core logic for the environment and neural policy network.
- `models/`: Saved model weights and optimization checkpoints.
- `notebooks/`: Experimental analysis and backtesting visualization.

## 🚀 Quick Start
```bash
python src/engine.py
```

---

**Lead Researcher:** Muhammad Zainurrahman  
**Framework:** PyTorch | NumPy | Pandas
