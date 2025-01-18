# Farm_RL (WIP)

Reinforcement learning project that creates and solves optimisation problems in the form of farming games.

## Usage

To clone the repo and fetch the required dependencies:

```bash
git clone https://github.com/FinlaySanders/Farm_RL.git
cd Farm_RL
uv sync
```

To train a model (will be saved in 'models'): 
```bash
uv run ppo.py
```
To train with Weights and Biases logging:
```bash
uv run ppo.py --track --wandb-project-name Farm_RL
```

To visualise a model in 'models':
```bash
uv run show.py --model example.pth
```

## Demo V1

Agent has partial observations (within the grey square) - decisions are made using an attention mechanism to weight crops, CNN for pathfinding, and a LSTM for memory.

https://github.com/user-attachments/assets/4dd4588e-c2de-4a09-8bc2-1f2b39fe6b67

## Demo V0

Agent has complete observations - decisions are made with a CNN.

https://github.com/user-attachments/assets/e0783d4d-0514-42c5-a3a3-0d44ae4cb367
