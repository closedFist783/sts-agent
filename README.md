# sts-agent

Reinforcement learning agent for Slay the Spire 2, trained via PPO using the [STS2AIAgent](https://github.com/CharTyr/STS2-Agent) mod HTTP API.

## Requirements
- Slay the Spire 2 with STS2AIAgent mod (exposes `http://localhost:8080`)
- Python 3.11+

```bash
pip install gymnasium stable-baselines3 requests numpy
```

## Usage
1. Launch StS2 with STS2AIAgent mod running
2. `python sts_env.py`

## Architecture
- **Observation:** 75-dim vector — player stats, 3 enemies × 4 features, 10 hand cards × 6 features
- **Actions:** Discrete(11) — play card 0–9 or end turn
- **Reward:** `enemy_hp_reduced - 2 * player_hp_lost` (+10 win, +50 elite, -20 death)
- **Algorithm:** PPO via Stable-Baselines3
