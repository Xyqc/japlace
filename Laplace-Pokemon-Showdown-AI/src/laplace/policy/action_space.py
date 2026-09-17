r"""Fixed-size action layout for the policy net, and the glue to/from `move_choice` strings.

`engine_search._pooled_policy` already speaks in `move_choice` strings ("switch <species>",
"<moveid>", "<moveid>-tera") -- see `_order_for_choice`. The policy net can't emit a variable-
length vocabulary of species/move ids directly (Random Battle draws from ~1000 mons), so
instead it emits logits over fixed SLOTS, mirroring Jaxcalibur's head layout:

    * 5 move slots (4 real moves + Struggle, matching poke-env's `battle.available_moves`
      ordering when Struggle is the only option), each with 2 logits: use / use+Tera.
      Jaxcalibur also gives Struggle "its own extra token"; we do the same rather than
      overloading move-slot 4.
    * 6 switch slots (own team order 0..5, i.e. `battle.team` insertion order), 1 logit each.

    16 logits total: [m0, m0_tera, m1, m1_tera, m2, m2_tera, m3, m3_tera,
                      struggle, struggle_tera(unused), s0, s1, s2, s3, s4, s5]

Struggle can't Tera and Tera can't be used mid-Struggle, so struggle_tera is always masked
out; it's kept as a slot rather than removed so the layout stays a flat 16 regardless of
what's legal this turn (a transformer with a variable-length action set is more machinery
than this phase needs).
"""

from poke_env.data import to_id_str

N_MOVE_SLOTS = 5          # 4 moves + Struggle
N_SWITCH_SLOTS = 6        # full team size in Random Battle
N_ACTIONS = 2 * N_MOVE_SLOTS + N_SWITCH_SLOTS   # 16

STRUGGLE_SLOT = N_MOVE_SLOTS - 1


def _move_slot_index(i, tera):
    return 2 * i + (1 if tera else 0)


def _switch_slot_index(i):
    return 2 * N_MOVE_SLOTS + i


def legal_actions(battle):
    """-> (mask: bool[16], slot_to_choice: {slot_index: move_choice string}).

    move_choice strings match exactly what `_pooled_policy` / `_order_for_choice` use, so a
    policy-net distribution over slots can be turned into a distribution over move_choice
    keys with no further translation, and pooled/visit-count targets from the existing search
    can be turned into slot targets for training.
    """
    mask = [False] * N_ACTIONS
    slot_to_choice = {}

    moves = list(battle.available_moves)
    can_tera = bool(getattr(battle, "can_tera", False))
    if moves and moves[0].id == "struggle":
        mask[_move_slot_index(STRUGGLE_SLOT, False)] = True
        slot_to_choice[_move_slot_index(STRUGGLE_SLOT, False)] = "struggle"
    else:
        for i, mv in enumerate(moves[:4]):
            idx = _move_slot_index(i, False)
            mask[idx] = True
            slot_to_choice[idx] = mv.id
            if can_tera:
                tidx = _move_slot_index(i, True)
                mask[tidx] = True
                slot_to_choice[tidx] = f"{mv.id}-tera"

    team_order = list(battle.team.values())
    switchable = {to_id_str(m.species) for m in battle.available_switches}
    for i, mon in enumerate(team_order[:N_SWITCH_SLOTS]):
        species = to_id_str(mon.species)
        if species in switchable:
            idx = _switch_slot_index(i)
            mask[idx] = True
            slot_to_choice[idx] = f"switch {species}"

    return mask, slot_to_choice


def choice_to_slot(choice, slot_to_choice):
    """Inverse lookup: move_choice string -> slot index, or None if not in this turn's map."""
    for slot, c in slot_to_choice.items():
        if c == choice:
            return slot
    return None


def targets_from_pooled(pooled, slot_to_choice):
    """Turn a `{move_choice: score}` dict (pooled visit shares) into a length-16 target
    distribution over slots, for policy-distillation training. Unmapped/illegal mass is
    dropped and the rest renormalized; an all-zero result means this turn isn't usable as a
    training example (e.g. every candidate expired between search and recording)."""
    target = [0.0] * N_ACTIONS
    total = 0.0
    for choice, score in pooled.items():
        slot = choice_to_slot(choice, slot_to_choice)
        if slot is not None and score > 0:
            target[slot] += score
            total += score
    if total > 0:
        target = [t / total for t in target]
    return target
