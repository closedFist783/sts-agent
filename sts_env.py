"""
Slay the Spire 2 RL Environment
Gymnasium-compatible environment for training RL agents on STS2
via the STS2AIAgent mod HTTP API.

Requires:
  - Slay the Spire 2 running with STS2AIAgent mod (https://github.com/CharTyr/STS2-Agent)
  - pip install gymnasium stable-baselines3 requests numpy

Usage:
  python sts_env.py
"""

import hashlib
import time

import gymnasium as gym
import numpy as np
import requests

# ── Config ────────────────────────────────────────────────────────────────────
API      = "http://127.0.0.1:8080"
OBS_SIZE = 75   # 3 player + 12 enemy (3×4) + 60 hand (10×6)

# Known elite enemy IDs — beating one gives +50 bonus reward
_ELITE_IDS = {
    "GREMLIN_NOB", "LAGAVULIN", "SENTRIES",
    "SLIME_BOSS", "THE_GUARDIAN", "HEXAGHOST",
    "CHAMP", "AUTOMATON", "COLLECTOR",
    "BOOK_OF_STABBING", "BYRDS", "GIANT_HEAD",
    "NEMESIS", "SHAKESPEARE", "TASKMASTER",
}

CARD_TYPES = {"Attack": 0, "Skill": 1, "Power": 2, "Status": 3, "Curse": 4}

# ── API helpers ───────────────────────────────────────────────────────────────

def _get_state() -> dict:
    return requests.get(f"{API}/state").json()["data"]

def _act(action: str, **params) -> dict:
    """Send an action. Converts numpy int types to Python int for JSON."""
    clean = {k: int(v) if hasattr(v, "item") else v for k, v in params.items()}
    r = requests.post(f"{API}/action", json={"action": action, **clean})
    return r.json()

# ── Observation encoding ──────────────────────────────────────────────────────

def _card_id_to_float(card_id: str) -> float:
    """Stable float encoding of a card ID via MD5 hash. Range 0.0–1.0."""
    h = int(hashlib.md5(card_id.encode()).hexdigest(), 16)
    return (h % 10000) / 10000.0

def _encode_obs(state: dict) -> np.ndarray:
    run    = state.get("run")    or {}
    cbt    = state.get("combat") or {}
    player = cbt.get("player")   or {}

    max_hp     = run.get("max_hp", 80)     or 80
    max_energy = run.get("max_energy", 3)  or 3

    # ── Player (3 features) ───────────────────────────────────────────────────
    player_hp    = run.get("current_hp", 80) / max_hp
    player_block = player.get("block", 0) / 50.0
    energy       = player.get("energy", 0) / max_energy

    # ── Enemies — up to 3 (12 features = 3 × 4) ──────────────────────────────
    enemies = cbt.get("enemies", [])
    enemy_feats: list[float] = []
    for i in range(3):
        if i < len(enemies):
            e = enemies[i]
            if e.get("is_alive"):
                ehp     = e.get("current_hp", 0) / max(e.get("max_hp", 1), 1)
                eblk    = e.get("block", 0) / 50.0
                intents = e.get("intents", [])
                is_atk  = 1.0 if any(x.get("intent_type") == "Attack" for x in intents) else 0.0
                total_dmg = sum(x.get("total_damage") or 0 for x in intents) / 50.0
                enemy_feats += [ehp, eblk, is_atk, total_dmg]
            else:
                enemy_feats += [0.0, 0.0, 0.0, 0.0]
        else:
            enemy_feats += [0.0, 0.0, 0.0, 0.0]

    # ── Hand — up to 10 cards (60 features = 10 × 6) ─────────────────────────
    hand = cbt.get("hand", [])
    hand_feats: list[float] = []
    for i in range(10):
        if i < len(hand):
            card     = hand[i]
            cid      = _card_id_to_float(card.get("card_id", ""))
            cost     = card.get("energy_cost", 0) / max_energy
            ctype    = CARD_TYPES.get(card.get("card_type", ""), 0) / 4.0
            upgraded = 1.0 if card.get("upgraded") else 0.0
            playable = 1.0 if card.get("playable") else 0.0
            # Primary damage or block value
            primary  = 0.0
            for dv in card.get("dynamic_values", []):
                if dv.get("name") in ("Damage", "Block"):
                    primary = dv.get("current_value", 0) / 50.0
                    break
            hand_feats += [cid, cost, ctype, upgraded, playable, primary]
        else:
            hand_feats += [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    obs = np.array([player_hp, player_block, energy] + enemy_feats + hand_feats,
                   dtype=np.float32)
    assert obs.shape == (OBS_SIZE,), f"Obs size mismatch: {obs.shape}"
    return obs

# ── Environment ───────────────────────────────────────────────────────────────

class STSCombatEnv(gym.Env):
    """
    STS2 combat-only RL environment.

    Observation: 75-dim vector (player stats, enemy stats × 3, hand cards × 10)
    Actions:     Discrete(11) — play card 0–9 or end turn (10)
    Reward:      enemy_hp_reduced - 2 * player_hp_lost
                 +10 for winning normal combat
                 +50 for winning elite combat
                 -20 for dying
    """
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(11)
        self._prev_player_hp  = 80
        self._prev_enemy_hp   = 0
        self._prev_enemies    = []  # for elite detection on done

    # ── Navigation ───────────────────────────────────────────────────────────

    def _skip_to_combat(self, max_steps: int = 400) -> dict:
        """Auto-navigate all non-combat screens until combat starts."""
        for _ in range(max_steps):
            state = _get_state()
            if state.get("in_combat") and state.get("available_actions"):
                return state
            if state.get("game_over"):
                _act("return_to_main_menu")
                time.sleep(0.5)
                continue

            actions = state.get("available_actions", [])
            screen  = state.get("screen", "")

            if screen == "GAME_OVER":
                _act("return_to_main_menu")
                time.sleep(0.5)
            elif screen == "MAIN_MENU":
                actions_on_menu = state.get("available_actions", [])
                if "abandon_run" in actions_on_menu:
                    # Abandon any in-progress run so we always start fresh
                    _act("abandon_run")
                    time.sleep(0.5)
                elif "continue_run" in actions_on_menu:
                    # Continue would resume mid-run — abandon it instead
                    _act("abandon_run")
                    time.sleep(0.5)
                else:
                    _act("open_character_select")
                    time.sleep(0.5)
            elif "embark" in actions:
                _act("embark")
            elif "choose_event_option" in actions:
                _act("choose_event_option", option_index=0)
            elif "collect_rewards_and_proceed" in actions:
                _act("collect_rewards_and_proceed")
            elif "skip_reward_cards" in actions:
                _act("skip_reward_cards")
            elif "choose_reward_card" in actions:
                opts = (state.get("reward") or {}).get("card_options", [])
                if opts:
                    _act("choose_reward_card", card_index=opts[0]["index"])
            elif "select_deck_card" in actions:
                sel   = state.get("selection") or {}
                cards = sel.get("cards", [])
                min_s = sel.get("min_select", 1)
                kind  = sel.get("kind", "")
                if min_s == 0:
                    # Optional selection (e.g. Neow card offer) — skip it
                    _act("proceed")
                elif cards:
                    _act("select_deck_card", card_index=cards[0]["index"])
            elif "proceed" in actions:
                _act("proceed")
            elif "choose_map_node" in actions:
                nodes = (state.get("map") or {}).get("available_nodes", [])
                if nodes:
                    monsters = [n for n in nodes if n.get("node_type") == "Monster"]
                    n = monsters[0] if monsters else nodes[0]
                    _act("choose_map_node", option_index=n["index"])
            elif "choose_rest_option" in actions:
                _act("choose_rest_option", option_id="rest")
            elif "confirm_modal" in actions:
                _act("confirm_modal")
            elif "dismiss_modal" in actions:
                _act("dismiss_modal")
            elif "close_main_menu_submenu" in actions:
                _act("close_main_menu_submenu")

            time.sleep(0.1)
        return _get_state()

    # ── Gym interface ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        state = self._skip_to_combat()
        run   = state.get("run")
        if not run:
            self._prev_player_hp = 80
            self._prev_enemy_hp  = 0
            self._prev_enemies   = []
            return np.zeros(OBS_SIZE, dtype=np.float32), {}
        cbt     = state.get("combat") or {}
        enemies = cbt.get("enemies", [])
        self._prev_player_hp = run["current_hp"]
        self._prev_enemy_hp  = sum(e.get("current_hp", 0) for e in enemies if e.get("is_alive"))
        self._prev_enemies   = enemies
        return _encode_obs(state), {}

    def step(self, action: int):
        state   = _get_state()
        cbt     = state.get("combat") or {}
        hand    = cbt.get("hand", [])
        action  = int(action)

        if action < 10 and action < len(hand):
            card = hand[action]
            if card.get("playable"):
                targets = card.get("valid_target_indices", [])
                if targets:
                    _act("play_card", card_index=action, target_index=targets[0])
                else:
                    _act("play_card", card_index=action)
            else:
                _act("end_turn")
        else:
            _act("end_turn")

        time.sleep(0.1)
        new_state   = _get_state()
        new_cbt     = new_state.get("combat") or {}
        new_enemies = new_cbt.get("enemies", [])
        run         = new_state.get("run") or {}

        new_player_hp = run.get("current_hp", self._prev_player_hp)
        new_enemy_hp  = sum(e.get("current_hp", 0) for e in new_enemies if e.get("is_alive"))

        enemy_reduced = max(0, self._prev_enemy_hp - new_enemy_hp)
        player_lost   = max(0, self._prev_player_hp - new_player_hp)
        reward = float(enemy_reduced - 2.0 * player_lost)

        self._prev_player_hp = new_player_hp
        self._prev_enemy_hp  = new_enemy_hp

        done = False
        if new_state.get("game_over"):
            done    = True
            reward -= 20.0
        elif not new_state.get("in_combat"):
            done = True
            # Elite bonus: check enemy IDs from previous combat state
            was_elite = any(
                e.get("enemy_id", "").upper() in _ELITE_IDS
                for e in self._prev_enemies
            )
            reward += 50.0 if was_elite else 10.0

        self._prev_enemies = new_enemies
        return _encode_obs(new_state), reward, done, False, {}

    def render(self):
        pass


# ── Training entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env

    env = STSCombatEnv()

    print("Checking environment...")
    check_env(env, warn=True)
    print("Environment OK\n")

    print("Training PPO agent...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device="cpu",
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
    )

    model.learn(total_timesteps=256)
    model.save("sts_ppo_model")
    print("\nModel saved → sts_ppo_model.zip")
