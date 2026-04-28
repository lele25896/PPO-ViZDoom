# PPO + LSTM for ViZDoom

A reinforcement learning agent that plays Doom using **Proximal Policy Optimization** with an **LSTM** memory unit, trained on two ViZDoom scenarios.

## Architecture

```
Input (4 × 84 × 84 grayscale frames)
    └─► CNN ──► Linear(3136 → 512)
                    └─► LSTMCell(512 → 256)
                              ├─► Actor  Linear(256 → N_actions) → action logits
                              └─► Critic Linear(256 → 1)         → state value V(s)
```

- **CNN** extracts spatial features from a stack of 4 grayscale frames
- **LSTM** carries temporal context across timesteps (enemy tracking, orientation memory)
- **Actor-Critic** shares the CNN+LSTM trunk; actor outputs action probabilities, critic estimates returns
- **GAE** (Generalized Advantage Estimation) for stable advantage computation

## Scenarios

### Basic
The agent faces a single stationary enemy directly ahead and must shoot it before the time limit.  
- 3 actions: `MOVE_LEFT`, `MOVE_RIGHT`, `ATTACK`  
- Trained from scratch · early stop at avg100 ≥ 86

### Defend the Center
The agent stands in a circular arena and is attacked by enemies spawning from all directions.  
- 7 actions: full movement + turn + attack  
- Trained from scratch · 3M steps

## Training Setup

| Parameter | Value |
|---|---|
| Algorithm | PPO |
| Network | CNN + LSTMCell |
| Parallel environments | 12 |
| Rollout steps per env | 512 |
| Samples per update | 6,144 |
| Learning rate | 2.5 × 10⁻⁴ |
| PPO clip | 0.2 |
| Discount γ | 0.99 |
| GAE λ | 0.95 |
| Mini-batch size | 256 |
| Frame stack | 4 |
| Frame size | 84 × 84 |

Multiprocessing uses `spawn` context (required on Windows) with workers defined in `doom_worker.py` so they can be imported by spawned subprocesses.

## Results

| Scenario | Avg reward (last 100 eps) |
|---|---|
| Basic | ~87 |
| Defend the center | ~5.1 kills |

## Files

| File | Description |
|---|---|
| `ppo_lstm_doom_multienv.ipynb` | Main notebook: model, training loop, evaluation |
| `doom_worker.py` | Worker process for parallel environments (spawn-safe) |
| `record_video.py` | Standalone script for recording gameplay videos |

## Requirements

```
torch
vizdoom
opencv-python
gymnasium
numpy
tqdm
```

Install:
```bash
pip install torch vizdoom opencv-python gymnasium numpy tqdm
```

## Usage

Run all cells in `ppo_lstm_doom_multienv.ipynb` in order.

To record videos after training:
```bash
python record_video.py --scenario basic   --model model_basic_multienv.pth  --output video_basic.mp4  --episodes 3
python record_video.py --scenario defend  --model model_defend_multienv.pth  --output video_defend.mp4 --episodes 3
```

## References

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [ViZDoom](https://vizdoom.farama.org/)
- [OpenAI Spinning Up — PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
