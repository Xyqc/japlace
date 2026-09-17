r"""Featurize (poke-engine State, poke-env Battle) into the policy net's three input blocks.

Reuses `laplace.value.value_features.featurize` for the "board" block wholesale (per the
brief: "reuse as much existing state parsing/serialization as practical" -- this is the same
368-dim vector the value net already trains on, so both nets see the position identically),
adds the Phase-2 history block (`policy.history`), and adds a per-action-slot feature block
so the transformer has more to go on for each of the 16 candidate actions than its bare
identity.

Board features come from an engine `State` (post-determinization, matches value_features'
own contract). Action features come from the live poke-env `Battle` instead, because they
need battle.available_moves/available_switches, which a determinized State doesn't carry in
a form worth re-deriving here -- current PP, boosted accuracy, etc. are already sitting on
the poke-env objects. This does mean action features are shared across all N determinized
worlds for a turn (only the opponent's identity is uncertain, and the policy purposefully
does not need to re-derive our own known move data N times).
"""

import numpy as np

from poke_env.data import GenData, to_id_str

from laplace.value.value_features import featurize as featurize_board, N_VALUE_FEATURES
from laplace.policy.action_space import N_ACTIONS, N_MOVE_SLOTS, N_SWITCH_SLOTS, legal_actions
from laplace.policy.history import N_HISTORY, EVENT_DIM, history_window

_TYPE_CHART = GenData.from_gen(9).type_chart

# per action-slot feature: valid flag, is_switch, is_tera_variant,
#   base_power/200, priority/5, category one-hot(3: phys/spec/status),
#   accuracy/100, pp_frac, best_effectiveness(0..4 log2-ish scaled), stab flag, hp_frac(switch)
ACTION_FEATURE_DIM = 13


def _move_action_features(mv, mon, opp_active):
    cat = getattr(mv, "category", None)
    cat_name = cat.name if cat is not None else "STATUS"
    cat_onehot = [1.0 if cat_name == c else 0.0 for c in ("PHYSICAL", "SPECIAL", "STATUS")]
    try:
        eff = mv.type.damage_multiplier(
            opp_active.type_1, opp_active.type_2, type_chart=_TYPE_CHART
        ) if (opp_active is not None and mv.type is not None) else 1.0
    except Exception:
        eff = 1.0
    stab = 1.0 if (mon is not None and mv.type is not None
                   and mv.type in (mon.type_1, mon.type_2)) else 0.0
    pp_frac = (mv.current_pp / mv.max_pp) if getattr(mv, "max_pp", 0) else 1.0
    return [
        1.0,                                   # valid
        0.0,                                   # is_switch
        0.0,                                   # is_tera_variant (overwritten by caller)
        min(mv.base_power, 200) / 200.0,
        max(min(getattr(mv, "priority", 0), 5), -5) / 5.0,
        *cat_onehot,
        (mv.accuracy if isinstance(mv.accuracy, (int, float)) else 100) / 100.0,
        pp_frac,
        min(eff, 4.0) / 4.0,
        stab,
        0.0,                                   # hp_frac, n/a for moves
    ]


def _switch_action_features(mon, opp_active):
    # Worst-case defensive matchup: the more dangerous of the opponent's (up to) two types
    # against the mon we'd be bringing in.
    eff = 1.0
    if opp_active is not None:
        try:
            opp_types = [t for t in (opp_active.type_1, opp_active.type_2) if t is not None]
            eff = max(
                (t.damage_multiplier(mon.type_1, mon.type_2, type_chart=_TYPE_CHART)
                 for t in opp_types),
                default=1.0,
            )
        except Exception:
            eff = 1.0
    return [
        1.0,                # valid
        1.0,                # is_switch
        0.0,                # is_tera_variant
        0.0, 0.0, 0.0, 0.0, 0.0,   # base_power/priority/category unused for switches
        0.0,                # accuracy n/a
        1.0,                # pp_frac n/a
        min(eff, 4.0) / 4.0,
        0.0,                # stab n/a
        mon.current_hp_fraction if mon.current_hp_fraction is not None else 0.0,
    ]


def action_features(battle):
    """-> (mask[16] bool, action_feats[16][ACTION_FEATURE_DIM] float, slot_to_choice dict)."""
    mask, slot_to_choice = legal_actions(battle)
    feats = [[0.0] * ACTION_FEATURE_DIM for _ in range(N_ACTIONS)]
    opp_active = battle.opponent_active_pokemon
    me = battle.active_pokemon

    moves = list(battle.available_moves)
    is_struggle = bool(moves) and moves[0].id == "struggle"
    for i, mv in enumerate(moves[:4] if not is_struggle else moves[:1]):
        base = _move_action_features(mv, me, opp_active)
        slot_use = 2 * (N_MOVE_SLOTS - 1 if is_struggle else i)
        feats[slot_use] = base
        tera_slot = slot_use + 1
        if mask[tera_slot]:
            tera_feats = list(base)
            tera_feats[2] = 1.0
            feats[tera_slot] = tera_feats

    team = list(battle.team.values())
    for i, mon in enumerate(team[:N_SWITCH_SLOTS]):
        slot = 2 * N_MOVE_SLOTS + i
        if mask[slot]:
            feats[slot] = _switch_action_features(mon, opp_active)

    return mask, feats, slot_to_choice


# Canonical order for the tera-type aux head's 18 output classes, matching the lowercase
# type-name strings `poke_engine_adapter.py` puts in PEPokemon.tera_type (e.g.
# `mon.tera_type.name.lower()`). Tera Stellar (a 19th, tera-only type) is deliberately NOT a
# class here -- it's rare in Random Battle rolls and, per the same graceful-degradation
# principle as an unmodeled item/ability, a mon that turns out to be Stellar just gets no
# posterior signal (neutral multiplier) rather than the model ever being asked to predict it.
TERA_TYPES = ("normal", "fire", "water", "electric", "grass", "ice", "fighting", "poison",
              "ground", "flying", "psychic", "bug", "rock", "ghost", "dragon", "dark",
              "steel", "fairy")


def expand_group_posterior(class_probs, groups):
    """A per-CLASS probability list (from item_head/ability_head, `_ITEM_FLAGS`-shaped
    groups where several concrete ids share one class because the net can't distinguish
    within a group) -> a per-ID dict, by handing every id in a group that class's
    probability. Ids belonging to no group (the trailing "other" class, or simply not
    covered by any group) are left OUT of the returned dict entirely -- see
    `poke_engine_adapter._posterior_multiplier`'s docstring for why a missing key means
    "neutral", which is the correct behaviour for "the model wasn't asked about this one"."""
    out = {}
    for group, prob in zip(groups, class_probs):
        for item_id in group:
            out[item_id] = float(prob)
    return out


def opponent_bench_species(battle):
    """Canonical slot order for the item/ability/tera aux heads (Phase 3): [active, then
    bench in `battle.opponent_team`'s own dict order], padded to 6 with None. This MUST
    match `poke_engine_adapter._opp_side`'s own ordering (`[active] + [bench mons in dict
    order]`) exactly, since it's how `engine_search._hidden_posterior_dict`'s per-slot
    predictions get matched back to a specific mon by species -- a mismatch here wouldn't
    error, it would just silently score the wrong Pokemon, which is worse than not scoring
    at all. If `_opp_side`'s ordering ever changes, change this to match."""
    active = battle.opponent_active_pokemon
    order = ([active] if active is not None else [])
    order += [m for m in battle.opponent_team.values() if m is not active]
    order = order[:6]   # matches net.N_OPP_BENCH; kept as a literal to avoid a features.py
                        # -> net.py import (net.py already imports FROM features.py)
    species = [to_id_str(m.species) for m in order]
    return species + [None] * (6 - len(species))


def build_inputs(state, battle, event_history, use_case_fix=True):
    """-> dict of numpy arrays ready for `net.PolicyNet.forward`:
        board:  float32[N_VALUE_FEATURES]
        history: float32[N_HISTORY, EVENT_DIM]
        action_feats: float32[N_ACTIONS, ACTION_FEATURE_DIM]
        action_mask: bool[N_ACTIONS]
        slot_to_choice: dict (not a tensor; carried through for decoding the output)
    """
    board = featurize_board(state, use_case_fix)
    hist = np.asarray(history_window(event_history, battle.turn), dtype=np.float32)
    mask, feats, slot_to_choice = action_features(battle)
    return {
        "board": np.asarray(board, dtype=np.float32),
        "history": hist,
        "action_feats": np.asarray(feats, dtype=np.float32),
        "action_mask": np.asarray(mask, dtype=bool),
        "slot_to_choice": slot_to_choice,
    }
