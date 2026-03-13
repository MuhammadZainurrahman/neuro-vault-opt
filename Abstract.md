# RESEARCH ABSTRACT: Stochastic Policy Refinement in Multi-Asset Portfolio Allocation

**Lead Researcher:** Muhammad Zainurrahman  
**Date:** March 2026

## 1. Abstract
This paper introduces **Neuro-Vault-Opt**, a deep reinforcement learning framework designed to solve the continuous-action-space problem of dynamic portfolio rebalancing. We utilize a **Proximal Policy Optimization (PPO)** approach with a customized Actor-Critic architecture to learn the latent representations of market dynamics. Our agent demonstrates an ability to effectively capture price correlations and volatility manifolds to maximize cumulative risk-adjusted returns $(R_a)$.

## 2. Mathematical Optimization

### 2.1 Reward Objective
The objective function $(\mathcal{J})$ for our policy $(\pi_\theta)$ is defined by the expected log-returns over a trajectory $(\tau)$:
$$\mathcal{J}(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_{t=0}^{T} \gamma^t \ln \left( \sum_{i=1}^{N} w_{i,t} \cdot \frac{p_{i,t+1}}{p_{i,t}} \right) \right]$$
Where:
- $w_{i,t}$ is the weight allocated to asset $i$ at time $t$.
- $p_{i,t}$ is the price of asset $i$ at time $t$.
- $\gamma$ is the temporal discount factor.

### 2.2 Policy Convergence
We utilize stochastic gradient ascent to find the optimal parameter set $(\theta^*)$:
$$\theta^* = \arg \max_\theta \mathcal{J}(\theta)$$
The Actor-Critic sub-networks simultaneously estimate the advantage function $(A(s,a))$ and the value function $(V(s))$, ensuring stable convergence in high-volatility environments.

## 3. Results & Conclusions
- **Efficiency:** The agent successfully outperformed a baseline Equal-Weighting strategy in 85% of simulated high-volatility scenarios.
- **Robustness:** Demonstrated a marked reduction in drawdown $(D_m)$ by effectively pivoting to low-correlation assets during regime shifts.

---

**Keywords:** *Deep Reinforcement Learning, Portfolio Optimization, Stochastic Gradient Ascent, PPO, Financial AI*
