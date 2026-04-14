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

# Observation layout (200 total):
#   8   player base       hp, block, energy, discard_ct_legacy, draw_ct_legacy,
#                         hand_size, draw_ct, discard_ct
#  20   player powers     10 powers × (id_hash, amount)
#  65   enemies           5 × 13 (hp, max_hp, block, alive, name_hash, is_elite,
#                                  intent_atk, intent_dmg, intent_hits, power_count,
#                                  vulnerable_stacks, weak_stacks, is_lethal)
#  30   enemy powers      5 enemies × 3 powers × (id_hash, amount)
#  60   hand cards        10 × 6 (card_hash, cost, type, upgraded, playable, primary_val)
#   3   run context       floor_pct, alive_enemies, hp_pct
#   9   potions           3 slots × (occupied, can_use, potion_id_hash)
#   5   deck composition  attack_ct, skill_ct, power_ct, status_ct, deck_size
OBS_SIZE = 203

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

# Persistent session for connection reuse (avoids socket exhaustion)
_session = requests.Session()
adapter  = requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=1)
_session.mount("http://", adapter)

def _get_state() -> dict:
    for attempt in range(5):
        try:
            return _session.get(f"{API}/state", timeout=10).json()["data"]
        except Exception:
            time.sleep(0.5 * (attempt + 1))
    return {}  # return empty state on total failure

def _act(action: str, **params) -> dict:
    clean = {k: int(v) if hasattr(v, "item") else v for k, v in params.items()}
    for attempt in range(5):
        try:
            r = _session.post(f"{API}/action", json={"action": action, **clean}, timeout=10)
            return r.json()
        except Exception:
            time.sleep(0.5 * (attempt + 1))
    return {"ok": False}

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
    hand       = cbt.get("hand", [])

    # ── Player base (8) ──────────────────────────────────────────────────────
    player_hp        = run.get("current_hp", 80) / max_hp
    player_block     = player.get("block", 0) / 50.0
    energy           = player.get("energy", 0) / max_energy
    # Legacy keys (may return 0 if game uses different field names — kept for obs shape)
    discard_ct_legacy = len(cbt.get("discard", [])) / 20.0
    draw_ct_legacy    = len(cbt.get("draw", [])) / 20.0
    hand_size         = len(hand) / 10.0
    # Correct keys — draw_pile / discard_pile or nested under piles.draw / piles.discard
    draw_ct    = min(len(cbt.get("draw_pile") or cbt.get("piles", {}).get("draw", [])), 40) / 40.0
    discard_ct = min(len(cbt.get("discard_pile") or cbt.get("piles", {}).get("discard", [])), 40) / 40.0

    # ── Player powers (20 = 10 × 2) ──────────────────────────────────────────
    pp_feats = _power_feats(player.get("powers", []), max_powers=10)

    # ── Pre-compute max damage from lowest-cost playable attack card ──────────
    attack_options = []
    for card in hand:
        if card.get("playable") and card.get("card_type") == "Attack":
            cost = card.get("energy_cost", 99)
            dmg  = 0
            for dv in card.get("dynamic_values", []):
                if dv.get("name") == "Damage":
                    dmg = dv.get("current_value", 0)
                    break
            attack_options.append((cost, dmg))
    max_lethal_dmg = 0
    if attack_options:
        min_cost      = min(c for c, _ in attack_options)
        max_lethal_dmg = max(d for c, d in attack_options if c == min_cost)

    # ── Enemies (65 = 5 × 13) + enemy powers (30 = 5 × 3 × 2) ──────────────
    enemies      = cbt.get("enemies", [])
    alive_count  = sum(1 for e in enemies if e.get("is_alive"))
    enemy_feats  = []
    epower_feats = []

    for i in range(5):
        if i < len(enemies):
            e        = enemies[i]
            alive    = 1.0 if e.get("is_alive") else 0.0
            raw_max  = max(e.get("max_hp", 1), 1)
            raw_hp   = e.get("current_hp", 0)
            ehp      = (raw_hp / raw_max) * alive
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
            # New: vulnerable / weak stacks from power list
            epowers     = e.get("powers", []) if alive else []
            vuln_norm  = next((min(abs(p.get("amount") or 0), 10) / 10.0
                              for p in e.get("powers", [])
                              if "VULNERABLE" in (p.get("power_id") or "").upper()), 0.0)
            weak_norm  = next((min(abs(p.get("amount") or 0), 10) / 10.0
                              for p in e.get("powers", [])
                              if "WEAK" in (p.get("power_id") or "").upper()), 0.0)
            # New: is_lethal flag
            is_lethal  = 1.0 if (alive and max_lethal_dmg > 0 and max_lethal_dmg >= raw_hp) else 0.0
            enemy_feats  += [ehp, emaxhp, eblk, alive, name_h, is_elite,
                             is_atk, tot_dmg, hits, pow_cnt,
                             vuln_norm, weak_norm, is_lethal]
            epower_feats += _power_feats(e.get("powers", []) if alive else [], max_powers=3)
        else:
            enemy_feats  += [0.0] * 13
            epower_feats += [0.0] * 6

    # ── Hand cards (60 = 10 × 6) ─────────────────────────────────────────────
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

    # ── Run context (3) ───────────────────────────────────────────────────────
    floor_pct  = run.get("floor", 1) / 55.0
    alive_frac = alive_count / 5.0
    # Screen context: what kind of screen is the model on?
    screen_h   = _card_id_to_float(state.get("screen", ""))
    event_h    = _card_id_to_float((state.get("event") or {}).get("event_id", ""))
    in_combat  = 1.0 if state.get("in_combat") else 0.0
    run_ctx    = [floor_pct, alive_frac, player_hp, screen_h, event_h, in_combat]

    # ── Potions (9 = 3 × 3) ──────────────────────────────────────────────────
    potions      = run.get("potions", [])
    potion_feats = []
    for i in range(3):
        if i < len(potions):
            pot      = potions[i]
            occupied = 1.0 if pot.get("potion_id", "EMPTY") not in ("", "EMPTY", None) else 0.0
            can_use  = 1.0 if pot.get("can_use", False) else 0.0
            pot_hash = _card_id_to_float(pot.get("potion_id", "")) if occupied else 0.0
            potion_feats += [occupied, can_use, pot_hash]
        else:
            potion_feats += [0.0, 0.0, 0.0]

    # ── Deck composition (5) ─────────────────────────────────────────────────
    deck          = run.get("deck", [])
    deck_attack_ct = min(sum(1 for c in deck if c.get("card_type") == "Attack"), 30) / 30.0
    deck_skill_ct  = min(sum(1 for c in deck if c.get("card_type") == "Skill"), 30) / 30.0
    deck_power_ct  = min(sum(1 for c in deck if c.get("card_type") == "Power"), 30) / 30.0
    deck_status_ct = min(sum(1 for c in deck if c.get("card_type") in ("Status", "Curse")), 10) / 10.0
    deck_size      = min(len(deck), 40) / 40.0
    deck_feats     = [deck_attack_ct, deck_skill_ct, deck_power_ct, deck_status_ct, deck_size]

    obs = np.array(
        [player_hp, player_block, energy, discard_ct_legacy, draw_ct_legacy, hand_size, draw_ct, discard_ct]
        + pp_feats
        + enemy_feats
        + epower_feats
        + hand_feats
        + run_ctx
        + potion_feats
        + deck_feats,
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
        self._prev_player_hp    = 80
        self._prev_player_blk   = 0
        self._prev_enemy_hp     = 0
        self._prev_enemies      = []
        self._enemy_attacking   = False  # for block efficiency reward
        self._turn_start_energy = 0      # energy at start of turn (for waste penalty)
        self._start_floor       = 1      # floor at episode start (for floor bonus)

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
                # Use MetaAgent for all events (not just Neow)
                _act("choose_event_option", option_index=_meta.choose_neow_option(state))
            elif "collect_rewards_and_proceed" in actions:
                _act("collect_rewards_and_proceed")
            elif "skip_reward_cards" in actions or "choose_reward_card" in actions:
                choice = _meta.choose_card_reward(state)
                if choice >= 0:
                    opts = (state.get("reward") or {}).get("card_options", [])
                    if opts and choice < len(opts):
                        _act("choose_reward_card", card_index=opts[choice]["index"])
                    else:
                        _act("skip_reward_cards")
                else:
                    _act("skip_reward_cards")
            elif "select_deck_card" in actions:
                sel   = state.get("selection") or {}
                cards = sel.get("cards", [])
                min_s = sel.get("min_select", 1)
                kind  = sel.get("kind", "")

                if not cards:
                    _act("proceed")
                else:
                    prompt = sel.get("prompt", "").lower()
                    is_remove    = "remove"    in kind or "remove"    in prompt
                    is_transform = "transform" in kind or "transform" in prompt
                    is_upgrade   = "upgrade"   in kind or "upgrade"   in prompt
                    is_exhaust   = "exhaust"   in kind or "exhaust"   in prompt

                    if is_exhaust:
                        # Exhaust from hand: select card if required, then confirm if available
                        if cards:
                            _act("select_deck_card", option_index=cards[0]["index"])
                            time.sleep(0.2)
                            post = _get_state()
                            if "confirm_selection" in (post.get("available_actions") or []):
                                _act("confirm_selection")
                        else:
                            # No cards to select, try confirm or proceed
                            post = _get_state()
                            acts = post.get("available_actions") or []
                            if "confirm_selection" in acts:
                                _act("confirm_selection")
                            elif "proceed" in acts:
                                _act("proceed")
                    elif is_remove:
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
                    best = _meta.choose_map_node(state, nodes)
                    _act("choose_map_node", option_index=nodes[best]["index"])
            elif "choose_rest_option" in actions:
                rest_opts = (state.get("rest") or {}).get("options", [])
                enabled   = [o for o in rest_opts if o.get("is_enabled")]
                preferred = _meta.choose_rest_option(state)
                target    = next((o for o in enabled if o.get("option_id") == preferred), None)
                opt_id    = target.get("option_id", "HEAL") if target else "HEAL"
                _act("choose_rest_option", option_id=opt_id)
            elif "confirm_selection" in actions and "select_deck_card" in actions:
                # Card selection with optional confirm — confirm with whatever is selected
                _act("confirm_selection")
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
            self._prev_player_hp   = 80
            self._prev_player_blk  = 0
            self._prev_enemy_hp    = 0
            self._prev_enemies     = []
            self._enemy_attacking  = False
            self._start_floor      = 1
            return np.zeros(OBS_SIZE, dtype=np.float32), {}
        cbt     = state.get("combat") or {}
        enemies = cbt.get("enemies", [])
        player  = cbt.get("player") or {}
        self._prev_player_hp    = run["current_hp"]
        self._prev_player_blk   = player.get("block", 0)
        self._prev_enemy_hp     = sum(e.get("current_hp", 0) for e in enemies if e.get("is_alive"))
        self._prev_enemies      = enemies
        self._turn_start_energy = player.get("energy", 0)
        self._start_floor       = run.get("floor", 1)
        self._enemy_attacking   = any(
            any(i.get("intent_type") == "Attack" for i in e.get("intents", []))
            for e in enemies if e.get("is_alive")
        )
        # Track starting relics (captures Neow choice at beginning of each run)
        if run.get("relics"):
            _tracker.record_neow_relics(run["relics"])
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
                # Track card play for analytics
                _tracker.record_card_play(card.get("card_id", "UNKNOWN"))
            else:
                _act("end_turn")
        else:
            _act("end_turn")

        time.sleep(0.02)

        # Handle mid-combat interrupts (card selections, modals that pause combat)
        for _ in range(20):
            mid         = _get_state()
            mid_actions = mid.get("available_actions", [])
            mid_screen  = mid.get("screen", "")
            # Break if we're back in active combat (can play cards/end turn) or done
            if mid.get("game_over") or not mid_actions:
                break
            if mid_screen == "COMBAT" and any(a in mid_actions for a in ("play_card", "end_turn")):
                break
            if "select_deck_card" in mid_actions:
                sel   = mid.get("selection") or {}
                cards = sel.get("cards", [])
                min_s = sel.get("min_select", 1)
                if cards:
                    _act("select_deck_card", option_index=cards[0]["index"])
                    time.sleep(0.1)
                    # Only confirm if that action is now available
                    post2 = _get_state()
                    if "confirm_selection" in (post2.get("available_actions") or []):
                        _act("confirm_selection")
                elif min_s == 0:
                    # No cards + optional — try proceed or confirm
                    post2 = _get_state()
                    acts2 = post2.get("available_actions") or []
                    if "confirm_selection" in acts2:
                        _act("confirm_selection")
                    elif "proceed" in acts2:
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

        # Kill bonus: capped at 5.0 per kill
        for i in range(len(self._prev_enemies)):
            prev_e = self._prev_enemies[i]
            if prev_e.get("current_hp", 0) > 0:      # was alive
                if i < len(new_enemies) and new_enemies[i].get("current_hp", 0) <= 0:
                    reward += min(prev_e.get("max_hp", 0) / 20.0, 5.0)

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
            # Floor progress bonus even on death
            new_floor = (new_state.get("run") or {}).get("floor", self._start_floor)
            reward   += (new_floor - self._start_floor) * 5
        elif not new_state.get("in_combat"):
            done = True
            was_elite = any(
                e.get("enemy_id", "").upper() in _ELITE_IDS
                for e in self._prev_enemies
            )
            reward += 50.0 if was_elite else 20.0
            # Floor progress bonus on win
            new_floor = run.get("floor", self._start_floor)
            reward   += (new_floor - self._start_floor) * 5

        self._prev_enemies = new_enemies
        return _encode_obs(new_state), reward, done, False, {}

    def render(self):
        pass


# ── MetaAgent (rule-based outside-combat decisions) ─────────────────────────

class MetaAgent:
    """Handles non-combat decisions with smart heuristics. Replaces all hardcoded choices."""

    def choose_rest_option(self, state: dict) -> str:
        run    = state.get("run") or {}
        hp_pct = run.get("current_hp", 80) / max(run.get("max_hp", 80), 1)
        return "HEAL" if hp_pct < 0.6 else "SMITH"

    def choose_card_reward(self, state: dict) -> int:
        """Returns card index to take, or -1 to skip."""
        opts = (state.get("reward") or {}).get("card_options", [])
        if not opts:
            return -1
        RARITY = {"Rare": 4, "Uncommon": 3, "Common": 2, "Starter": 0}
        CTYPE  = {"Power": 3, "Attack": 2, "Skill": 1}
        best_i, best_s = 0, -1
        for i, card in enumerate(opts):
            s = RARITY.get(card.get("rarity", "Common"), 1) + CTYPE.get(card.get("card_type", "Skill"), 1)
            if card.get("card_type") in ("Status", "Curse"):
                s = -10
            if s > best_s:
                best_s, best_i = s, i
        return best_i if best_s > 2 else -1

    def choose_map_node(self, state: dict, nodes: list) -> int:
        run    = state.get("run") or {}
        hp_pct = run.get("current_hp", 80) / max(run.get("max_hp", 80), 1)
        PRIORITY = {
            "Elite":    5 if hp_pct > 0.6 else -1,
            "Monster":  3,
            "Rest":     4 if hp_pct < 0.5 else 2,
            "Shop":     1,
            "Event":    2,
            "Treasure": 4,
        }
        best_i, best_s = 0, -1
        for i, node in enumerate(nodes):
            s = PRIORITY.get(node.get("node_type", ""), 1)
            if s > best_s:
                best_s, best_i = s, i
        return best_i

    def choose_neow_option(self, state: dict) -> int:
        """Smart event option picker for all events (Neow, Unknown rooms, etc.)."""
        opts = (state.get("event") or {}).get("options", [])
        if not opts:
            return 0
        avail = [o for o in opts if not o.get("is_locked", False) and not o.get("will_kill_player", False)]
        if not avail:
            avail = opts  # fallback: all options

        def score(o):
            desc = (o.get("description") or "").lower()
            s = 0
            if "lose" in desc and "max hp" in desc:   s -= 6
            if "lose" in desc and " hp" in desc:       s -= 3
            if "curse" in desc:                        s -= 3
            if "gold" in desc and "lose" not in desc:  s += 2
            if "card" in desc and "lose" not in desc:  s += 2
            if "relic" in desc and "lose" not in desc: s += 3
            if "heal" in desc:                         s += 1
            return s

        avail.sort(key=score, reverse=True)
        return avail[0].get("index", 0)


_meta = MetaAgent()

# ── Weight migration ──────────────────────────────────────────────────────────

def migrate_model_weights(old_path: str, new_env, new_obs_size: int):
    """Load old model, zero-pad input layer to match new obs size, return new model."""
    import torch
    from stable_baselines3 import PPO as _PPO
    print(f"Migrating weights from {old_path} to OBS_SIZE={new_obs_size}...")
    old = _PPO.load(old_path, device="cpu")
    old_size = old.observation_space.shape[0]
    if old_size == new_obs_size:
        print("  Obs sizes match — reusing directly")
        old.set_env(new_env)
        return old
    new = _PPO("MlpPolicy", new_env, verbose=1, device="cpu",
               learning_rate=3e-4, n_steps=256, batch_size=64, n_epochs=4, gamma=0.99)
    old_sd, new_sd = old.policy.state_dict(), new.policy.state_dict()
    for key in new_sd:
        if key not in old_sd:
            continue
        ot, nt = old_sd[key], new_sd[key]
        if ot.shape == nt.shape:
            new_sd[key] = ot.clone()
        elif key.endswith(".weight") and len(ot.shape) == 2 and ot.shape[0] == nt.shape[0] and ot.shape[1] < nt.shape[1]:
            pad = torch.zeros_like(nt)
            pad[:, :ot.shape[1]] = ot
            new_sd[key] = pad
            print(f"  Padded {key}: {list(ot.shape)} → {list(nt.shape)}")
        else:
            new_sd[key] = ot.clone()
    new.policy.load_state_dict(new_sd)
    print(f"  Migration done: {old_size} → {new_obs_size} dims")
    return new


# ── Analytics tracker ─────────────────────────────────────────────────────────

class STSTracker:
    """Lightweight analytics tracker for STS2 training sessions."""

    def __init__(self):
        self.card_play_counts  = {}   # card_id  → count
        self.neow_relic_counts = {}   # relic_id → count

    def record_card_play(self, card_id: str):
        """Increment play count for a card."""
        self.card_play_counts[card_id] = self.card_play_counts.get(card_id, 0) + 1

    # Starter relics every character always has — not worth tracking
    _STARTER_RELICS = {"BURNING_BLOOD", "PURE_WATER", "CRACKED_CORE", "RING_OF_THE_SNAKE"}

    def record_neow_relics(self, relics: list):
        """Record non-starter relics (Neow picks and run gains)."""
        for relic in relics:
            rid = relic.get("relic_id", "UNKNOWN")
            if rid not in self._STARTER_RELICS:
                self.neow_relic_counts[rid] = self.neow_relic_counts.get(rid, 0) + 1

    def print_summary(self):
        print("\n── Analytics ───────────────────────────────────────────")
        print("  Top 5 most played cards:")
        sorted_cards = sorted(self.card_play_counts.items(), key=lambda x: x[1], reverse=True)
        for rank, (cid, cnt) in enumerate(sorted_cards[:5], 1):
            print(f"    {rank}. {cid}: {cnt}")
        if not sorted_cards:
            print("    (none recorded)")
        print("  Top 5 Neow relics chosen (floor 1):")
        sorted_relics = sorted(self.neow_relic_counts.items(), key=lambda x: x[1], reverse=True)
        for rank, (rid, cnt) in enumerate(sorted_relics[:5], 1):
            print(f"    {rank}. {rid}: {cnt}")
        if not sorted_relics:
            print("    (none recorded)")
        print("─" * 50)


# Module-level tracker instance — shared by all env instances
_tracker = STSTracker()


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

    class CheckpointCallback(BaseCallback):
        def __init__(self, save_every=2500, save_dir="models"):
            super().__init__()
            self.save_every = save_every
            self.save_dir   = save_dir
            self._last_save = 0

        def _on_step(self):
            if self.num_timesteps - self._last_save >= self.save_every:
                import time as _t
                ts   = _t.strftime("%Y%m%d_%H%M%S")
                path = os.path.join(self.save_dir, f"sts_checkpoint_{ts}_step{self.num_timesteps}")
                self.model.save(path)
                print(f"  [Checkpoint] Saved → {path}.zip")
                self._last_save = self.num_timesteps
            return True

    class RolloutCallback(BaseCallback):
        """Prints improvement vs previous rollout after each rollout ends."""
        def __init__(self):
            super().__init__()
            self._prev_mean = None

        def _on_step(self):
            return True

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

    # Find best model to load: main > latest checkpoint > fresh
    def _find_best_model():
        if os.path.exists("sts_ppo_model.zip"):
            return "sts_ppo_model"
        # Find latest checkpoint by step number
        checkpoints = sorted(
            [f for f in os.listdir(MODELS_DIR) if f.startswith("sts_checkpoint_") and f.endswith(".zip")],
            key=lambda f: int(f.split("_step")[-1].replace(".zip", "")) if "_step" in f else 0,
            reverse=True
        ) if os.path.exists(MODELS_DIR) else []
        if checkpoints:
            return os.path.join(MODELS_DIR, checkpoints[0].replace(".zip", ""))
        return None

    best_model = _find_best_model()
    if best_model:
        print(f"Loading model: {best_model}.zip (migrating weights if obs size changed)...")
        try:
            model = migrate_model_weights(best_model, env, OBS_SIZE)
            print("Loaded. Continuing from previous weights.\n")
        except Exception as e:
            print(f"Migration failed ({e}) — starting fresh.\n")
            model = PPO("MlpPolicy", env, verbose=1, device="cpu",
                        learning_rate=3e-4, n_steps=256, batch_size=64, n_epochs=4, gamma=0.99)
    else:
        print("No existing model — starting fresh.\n")
        model = PPO("MlpPolicy", env, verbose=1, device="cpu",
                    learning_rate=3e-4, n_steps=256, batch_size=64, n_epochs=4, gamma=0.99)

    print("Training PPO agent...")
    TIMESTEPS = 57000
    t_start   = _time.time()
    model.learn(total_timesteps=TIMESTEPS, callback=[RolloutCallback(), CheckpointCallback()], reset_num_timesteps=False)
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

    # Print analytics
    _tracker.print_summary()
