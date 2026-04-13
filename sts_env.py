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

# ── Config ─────────────────────────────────────────────────────────────────────
API = "http://127.0.0.1:8080"

# Observation layout (161 total):
#   3   player base       hp, block, energy
#  20   player powers     10 powers × (id_hash, amount)
#  50   enemies           5 × 10 (hp, max_hp, block, alive, name_hash, is_elite,
#                                 intent_atk, intent_dmg, intent_hits, power_count)
#  30   enemy powers      5 enemies × 3 powers × (id_hash, amount)
#  60   hand cards        10 × 6 (card_hash, cost, type, upgraded, playable, primary_val)
#   3   run context       floor_pct, alive_enemies, hp_pct
OBS_SIZE = 166

# Action space: MultiDiscrete([11, 5])
#   action[0]: 0-9 = play card at hand index, 10 = end turn
#   action[1]: 0-4 = target enemy index (ignored if card doesn't need target or action=10)
#   STS2 can have up to 5 enemies (Sentries, slime splits, etc.)
N_CARD_ACTIONS   = 11
N_TARGET_CHOICES = 5

# Known elite enemy IDs — beating one gives +50 bonus reward
_ELITE_IDS = {
    "GREMLIN_NOB", "LAGAVULIN", "SENTRIES",
    "SLIME_BOSS", "THE_GUARDIAN", "HEXAGHOST",
    "CHAMP", "AUTOMATON", "COLLECTOR",
    "BOOK_OF_STABBING", "BYRDS", "GIANT_HEAD",
    "NEMESIS", "SHAKESPEARE", "TASKMASTER",
}

CARD_TYPES = {"Attack": 0, "Skill": 1, "Power": 2, "Status": 3, "Curse": 4}

# ── API helpers ────────────────────────────────────────────────────────────────

def _get_state() -> dict:
    return requests.get(f"{API}/state").json()["data"]

def _act(action: str, **params) -> dict:
    clean = {k: int(v) if hasattr(v, "item") else v for k, v in params.items()}
    r = requests.post(f"{API}/action", json={"action": action, **clean})
    return r.json()

# ── Observation encoding ───────────────────────────────────────────────────────

def _card_id_to_float(card_id: str) -> float:
    h = int(hashlib.md5(card_id.encode()).hexdigest(), 16)
    return (h % 10000) / 10000.0

def _power_feats(powers: list, max_powers: int) -> list:
    feats = []
    for i in range(max_powers):
        if i < len(powers):
            p = powers[i]
            pid    = _card_id_to_float(p.get("power_id", ""))
            amount = min(abs(p.get("amount") or 0), 99) / 99.0
            feats += [pid, amount]
        else:
            feats += [0.0, 0.0]
    return feats

def _encode_obs(state: dict) -> np.ndarray:
    run    = state.get("run")    or {}
    cbt    = state.get("combat") or {}
    player = cbt.get("player")   or {}

    max_hp     = run.get("max_hp", 80)    or 80
    max_energy = run.get("max_energy", 3) or 3

    # ── Player base (3) ──────────────────────────────────────────────────────
    player_hp    = run.get("current_hp", 80) / max_hp
    player_block = player.get("block", 0) / 50.0
    energy       = player.get("energy", 0) / max_energy

    # ── Player powers (20 = 10 × 2) ──────────────────────────────────────────
    pp_feats = _power_feats(player.get("powers", []), max_powers=10)

    # ── Enemies (27 = 3 × 9) + enemy powers (18 = 3 × 3 × 2) ────────────────
    enemies      = cbt.get("enemies", [])
    alive_count  = sum(1 for e in enemies if e.get("is_alive"))
    enemy_feats  = []
    epower_feats = []

    for i in range(5):
        if i < len(enemies):
            e        = enemies[i]
            alive    = 1.0 if e.get("is_alive") else 0.0
            raw_max  = max(e.get("max_hp", 1), 1)
            ehp      = (e.get("current_hp", 0) / raw_max) * alive
            emaxhp   = min(raw_max, 500) / 500.0
            eblk     = e.get("block", 0) / 50.0 * alive
            name_h   = _card_id_to_float(e.get("enemy_id", ""))
            is_elite = 1.0 if e.get("enemy_id", "").upper() in _ELITE_IDS else 0.0
            intents  = e.get("intents", []) if alive else []
            atk_ints = [x for x in intents if x.get("intent_type") == "Attack"]
            is_atk   = 1.0 if atk_ints else 0.0
            tot_dmg  = min(sum(x.get("total_damage") or 0 for x in atk_ints), 99) / 99.0
            hits     = min(sum(x.get("hits") or 0 for x in atk_ints), 9) / 9.0
            pow_cnt  = min(len(e.get("powers", [])), 10) / 10.0
            enemy_feats  += [ehp, emaxhp, eblk, alive, name_h, is_elite, is_atk, tot_dmg, hits, pow_cnt]
            epower_feats += _power_feats(e.get("powers", []) if alive else [], max_powers=3)
        else:
            enemy_feats  += [0.0] * 10
            epower_feats += [0.0] * 6

    # ── Hand cards (60 = 10 × 6) ─────────────────────────────────────────────
    hand       = cbt.get("hand", [])
    hand_feats = []
    for i in range(10):
        if i < len(hand):
            card     = hand[i]
            cid      = _card_id_to_float(card.get("card_id", ""))
            cost     = card.get("energy_cost", 0) / max_energy
            ctype    = CARD_TYPES.get(card.get("card_type", ""), 0) / 4.0
            upgraded = 1.0 if card.get("upgraded") else 0.0
            playable = 1.0 if card.get("playable") else 0.0
            primary  = 0.0
            for dv in card.get("dynamic_values", []):
                if dv.get("name") in ("Damage", "Block"):
                    primary = min(dv.get("current_value", 0), 99) / 99.0
                    break
            hand_feats += [cid, cost, ctype, upgraded, playable, primary]
        else:
            hand_feats += [0.0] * 6

    # ── Run context (4) ───────────────────────────────────────────────────────
    floor_pct  = run.get("floor", 1) / 55.0
    alive_frac = alive_count / 5.0
    run_ctx    = [floor_pct, alive_frac, player_hp]

    obs = np.array(
        [player_hp, player_block, energy]
        + pp_feats
        + enemy_feats
        + epower_feats
        + hand_feats
        + run_ctx,
        dtype=np.float32
    )
    assert obs.shape[0] == OBS_SIZE, f"Obs mismatch: got {obs.shape[0]}, want {OBS_SIZE}"
    return obs

# ── Environment ────────────────────────────────────────────────────────────────

class STSCombatEnv(gym.Env):
    """
    STS2 combat-only RL environment.

    Obs:    132-dim vector (player stats+powers, 3 enemies with powers, 10 hand cards, run context)
    Actions: Discrete(11) — play card 0-9 or end turn (10)
    Reward:  enemy_hp_reduced - 2*player_hp_lost + block_efficiency
             +10 win, +50 elite win, -20 death
    """
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32
        )
        # MultiDiscrete: [card_action (0-10), target (0-2)]
        self.action_space = gym.spaces.MultiDiscrete([N_CARD_ACTIONS, N_TARGET_CHOICES])
        self._prev_player_hp   = 80
        self._prev_player_blk  = 0
        self._prev_enemy_hp    = 0
        self._prev_enemies     = []
        self._enemy_attacking  = False  # for block efficiency reward
        self._turn_start_energy = 0     # energy at start of turn (for waste penalty)

    # ── Navigation ────────────────────────────────────────────────────────────

    def _skip_to_combat(self, max_steps: int = 400) -> dict:
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
                menu_actions = state.get("available_actions", [])
                if "abandon_run" in menu_actions or "continue_run" in menu_actions:
                    _act("abandon_run")
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

                if not cards:
                    _act("proceed")
                else:
                    prompt = sel.get("prompt", "").lower()
                    is_remove   = "remove" in kind or "remove" in prompt
                    is_transform = "transform" in kind or "transform" in prompt
                    is_upgrade   = "upgrade" in kind or "upgrade" in prompt

                    if is_remove:
                        target = (
                            next((c for c in cards if "STRIKE" in c.get("card_id", "").upper()), None)
                            or next((c for c in cards if "DEFEND" in c.get("card_id", "").upper()), None)
                            or cards[0]
                        )
                    elif is_transform:
                        target = (
                            next((c for c in cards if "STRIKE" in c.get("card_id", "").upper()), None)
                            or next((c for c in cards if "DEFEND" in c.get("card_id", "").upper()), None)
                            or cards[0]
                        )
                    elif is_upgrade:
                        target = (
                            next((c for c in cards if "BASH" in c.get("card_id", "").upper()), None)
                            or next((c for c in cards if "STRIKE" in c.get("card_id", "").upper()), None)
                            or next((c for c in cards if "DEFEND" in c.get("card_id", "").upper()), None)
                            or cards[0]
                        )
                    else:
                        target = cards[0]
                    _act("select_deck_card", option_index=target["index"])

                # Confirm if needed
                time.sleep(0.2)
                post = _get_state()
                if "confirm_selection" in (post.get("available_actions") or []):
                    _act("confirm_selection")

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

    # ── Gym interface ──────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        state = self._skip_to_combat()
        run   = state.get("run")
        if not run:
            self._prev_player_hp  = 80
            self._prev_player_blk = 0
            self._prev_enemy_hp   = 0
            self._prev_enemies    = []
            self._enemy_attacking = False
            return np.zeros(OBS_SIZE, dtype=np.float32), {}
        cbt     = state.get("combat") or {}
        enemies = cbt.get("enemies", [])
        player  = cbt.get("player") or {}
        self._prev_player_hp   = run["current_hp"]
        self._prev_player_blk  = player.get("block", 0)
        self._prev_enemy_hp    = sum(e.get("current_hp", 0) for e in enemies if e.get("is_alive"))
        self._prev_enemies     = enemies
        self._turn_start_energy = player.get("energy", 0)
        self._enemy_attacking  = any(
            any(i.get("intent_type") == "Attack" for i in e.get("intents", []))
            for e in enemies if e.get("is_alive")
        )
        return _encode_obs(state), {}

    def step(self, action):
        state   = _get_state()
        cbt     = state.get("combat") or {}
        hand    = cbt.get("hand", [])
        enemies = cbt.get("enemies", [])

        # Unpack MultiDiscrete action [card_idx, target_idx]
        card_action  = int(action[0])
        target_pref  = int(action[1])  # preferred target index (0-2)

        if card_action < 10 and card_action < len(hand):
            card = hand[card_action]
            if card.get("playable"):
                valid_targets = card.get("valid_target_indices", [])
                if valid_targets:
                    # Use agent's preferred target if valid, else first valid target
                    target = target_pref if target_pref in valid_targets else valid_targets[0]
                    _act("play_card", card_index=card_action, target_index=target)
                else:
                    _act("play_card", card_index=card_action)
            else:
                _act("end_turn")
        else:
            _act("end_turn")

        time.sleep(0.02)

        # Handle mid-combat interrupts
        for _ in range(20):
            mid         = _get_state()
            mid_actions = mid.get("available_actions", [])
            if mid.get("in_combat") or mid.get("game_over") or not mid_actions:
                break
            if "select_deck_card" in mid_actions:
                sel   = mid.get("selection") or {}
                cards = sel.get("cards", [])
                if cards:
                    _act("select_deck_card", option_index=cards[0]["index"])
                else:
                    _act("proceed")
            elif "confirm_modal" in mid_actions:
                _act("confirm_modal")
            elif "dismiss_modal" in mid_actions:
                _act("dismiss_modal")
            elif "proceed" in mid_actions:
                _act("proceed")
            else:
                break
            time.sleep(0.1)

        new_state   = _get_state()
        new_cbt     = new_state.get("combat") or {}
        new_enemies = new_cbt.get("enemies", [])
        new_player  = new_cbt.get("player") or {}
        run         = new_state.get("run") or {}

        new_player_hp  = run.get("current_hp", self._prev_player_hp)
        new_player_blk = new_player.get("block", 0)
        new_enemy_hp   = sum(e.get("current_hp", 0) for e in new_enemies if e.get("is_alive"))

        # Base reward
        enemy_reduced = max(0, self._prev_enemy_hp - new_enemy_hp)
        player_lost   = max(0, self._prev_player_hp - new_player_hp)
        reward        = float(enemy_reduced - 2.0 * player_lost)

        # Energy waste penalty: -5 per unspent energy when turn ends
        # Detect turn end: action was end_turn OR new turn started (energy refilled)
        new_energy = new_player.get("energy", 0)
        new_max_energy = run.get("max_energy", 3) or 3
        if card_action == 10:  # explicit end_turn
            leftover = self._turn_start_energy
            reward -= leftover * 5.0
        # Track energy for next turn
        if new_energy >= new_max_energy - 1:  # turn just reset
            self._turn_start_energy = new_energy
        else:
            self._turn_start_energy = new_energy

        # Block efficiency bonus: reward gaining block when enemy was going to attack
        if self._enemy_attacking:
            block_gained = max(0, new_player_blk - self._prev_player_blk)
            reward += block_gained * 0.3  # small bonus for smart blocking

        self._prev_player_hp  = new_player_hp
        self._prev_player_blk = new_player_blk
        self._prev_enemy_hp   = new_enemy_hp

        # Track if enemy is attacking next turn for block efficiency
        self._enemy_attacking = any(
            any(i.get("intent_type") == "Attack" for i in e.get("intents", []))
            for e in new_enemies if e.get("is_alive")
        )

        done = False
        if new_state.get("game_over"):
            done    = True
            reward -= 20.0
        elif not new_state.get("in_combat"):
            done = True
            was_elite = any(
                e.get("enemy_id", "").upper() in _ELITE_IDS
                for e in self._prev_enemies
            )
            reward += 50.0 if was_elite else 10.0

        self._prev_enemies = new_enemies
        return _encode_obs(new_state), reward, done, False, {}

    def render(self):
        pass


# ── Training entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    import json
    import time as _time
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.callbacks import BaseCallback

    HISTORY_FILE = "training_history.json"
    MODELS_DIR   = "models"
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Load previous run stats for comparison
    history = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE) as f:
            history = json.load(f)
    prev = history[-1] if history else None

    class RolloutCallback(BaseCallback):
        """Prints improvement vs previous rollout after each rollout ends."""
        def __init__(self):
            super().__init__()
            self._prev_mean = None

        def _on_rollout_end(self):
            rews = self.model.env.envs[0].get_episode_rewards()
            if len(rews) < 2:
                return True
            recent = float(np.mean(rews[-3:]))
            if self._prev_mean is not None:
                delta = recent - self._prev_mean
                sign  = "+" if delta >= 0 else ""
                print(f"  └ Rollout avg (last 3 eps): {recent:.1f}  ({sign}{delta:.1f} vs previous)")
            self._prev_mean = recent
            return True

    env = STSCombatEnv()

    print("Checking environment...")
    check_env(env, warn=True)
    print("Environment OK\n")

    # Load existing model if available, else create new
    if os.path.exists("sts_ppo_model.zip"):
        print("Loading existing model for continued training...")
        model = PPO.load("sts_ppo_model", env=env)
        print("Loaded. Continuing from previous weights.\n")
    else:
        print("No existing model — starting fresh.\n")
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

    print("Training PPO agent...")
    TIMESTEPS = 2000
    t_start   = _time.time()
    model.learn(total_timesteps=TIMESTEPS, callback=RolloutCallback(), reset_num_timesteps=False)
    t_end   = _time.time()
    elapsed = t_end - t_start

    # Versioned + latest save
    ts_str   = _time.strftime("%Y%m%d_%H%M%S")
    ver_path = os.path.join(MODELS_DIR, f"sts_ppo_{ts_str}")
    model.save(ver_path)
    model.save("sts_ppo_model")
    print(f"\nModels saved:\n  Latest : sts_ppo_model.zip\n  Version: {ver_path}.zip")

    # ── Post-training summary ─────────────────────────────────────────────────
    monitor       = model.env.envs[0]
    ep_rewards    = monitor.get_episode_rewards()
    ep_lengths    = monitor.get_episode_lengths()
    secs_per_step = elapsed / TIMESTEPS if TIMESTEPS else 0

    if ep_rewards:
        wins     = [r for r in ep_rewards if r > 0]
        avg_rew  = float(np.mean(ep_rewards))
        win_rate = 100 * len(wins) / len(ep_rewards)

        print("\n" + "=" * 48)
        print("  TRAINING SUMMARY")
        print("=" * 48)
        print(f"  Timesteps           {TIMESTEPS:>8}")
        print(f"  Total time          {elapsed:>7.1f}s  ({elapsed/60:.1f} min)")
        print(f"  Time per step       {secs_per_step:>8.3f}s")
        if secs_per_step > 0:
            print(f"  Steps per second    {1/secs_per_step:>8.1f}")
        print(f"  Episodes            {len(ep_rewards):>8}")
        print(f"  Avg reward          {avg_rew:>8.1f}")
        print(f"  Best episode        {max(ep_rewards):>8.1f}")
        print(f"  Worst episode       {min(ep_rewards):>8.1f}")
        print(f"  Std dev             {np.std(ep_rewards):>8.1f}")
        print(f"  Avg length (steps)  {np.mean(ep_lengths):>8.1f}")
        print(f"  Wins (reward > 0)   {len(wins):>5} / {len(ep_rewards)}")
        print(f"  Win rate            {win_rate:>7.0f}%")
        if prev:
            d_rew  = avg_rew - prev["avg_reward"]
            d_win  = win_rate - prev["win_rate"]
            sr, sw = ("+" if d_rew >= 0 else ""), ("+" if d_win >= 0 else "")
            print(f"  ── vs last run ─────────────────────────────")
            print(f"  Reward change       {sr}{d_rew:>7.1f}  ({prev['avg_reward']:.1f} → {avg_rew:.1f})")
            print(f"  Win rate change     {sw}{d_win:>6.0f}%  ({prev['win_rate']:.0f}% → {win_rate:.0f}%)")
        print("=" * 48)

        history.append({
            "timestamp":   ts_str,
            "timesteps":   TIMESTEPS,
            "avg_reward":  avg_rew,
            "best":        float(max(ep_rewards)),
            "win_rate":    win_rate,
            "episodes":    len(ep_rewards),
            "secs_step":   secs_per_step,
        })
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)
        print(f"  History → {HISTORY_FILE}")
    else:
        print(f"\nNo completed episodes in {elapsed:.1f}s — increase total_timesteps")
