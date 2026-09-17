r"""Generate (board/history/action features, policy target, value target) training data for
the Phase-1 policy net, by self-distillation from the EXISTING guarded search.

This is not full self-play RL (Jaxcalibur's PPO loop) -- it's the practical first step: two
EnginePlayers battle each other on the local server, and at every decision we record what the
pooled MCTS + guards + mixed-root pipeline ultimately decided (the "decision" event's final,
post-guard `ranked` list, turned into a target distribution over the 16 action slots via
`policy.action_space.targets_from_pooled`), labelled afterward with the game's actual
win/loss. The policy net is trained to imitate this guard-corrected search output -- a
teacher that already embodies the absorb/futility/gamble/deadlock guards this project asked
to keep, rather than training against raw unguarded MCTS visits.

This deliberately mirrors gen_value_data.py's structure and sharding so both nets can be
generated from the same kind of run and eyeballed against each other.

Item/ability/Tera targets (Phase 3) ARE collected, via a form of hindsight labelling that's
standard for hidden-information prediction in imperfect-info games (the same idea as
training a poker/Stratego hidden-info predictor against what eventually got shown down):
at each decision we snapshot which species occupied which of the 6 canonical bench slots
(`policy.features.opponent_bench_species`), and after the game ends we look up whatever
poke-env ended up actually learning about that species' item/ability/tera -- which is
STRICTLY MORE than what was known at collection time, because reveals accumulate over the
game. A slot whose item/ability/tera was NEVER revealed (very common for tera specifically,
since it only reveals on an actual terastallization) is left unlabelled -- see
`train_policy.py`'s masked loss, not zero-filled or guessed.

Opponent-ACTION targets (Phase 4) are still NOT collected -- see net.py's docstring on why
the current `opp_action_head` layout has no established correspondence to a specific
opponent's actual moves, which a training target needs and inference-time slot-matching
doesn't currently have either.

    python -m laplace.cli.gen_policy_data --battles 250 --workers 10
"""

import argparse
import asyncio
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

from laplace import paths

BATTLE_FORMAT = "gen9randombattle"
DATA_DIR = paths.POLICY_DATA_DIR
N_OPP_BENCH = 6


def _class_index(value, groups):
    """value (an item/ability id) -> index into `groups` (an _ITEM_FLAGS/_ABILITY_FLAGS-
    shaped tuple of tuples), or -1 if value doesn't belong to any group (either it's a
    real id outside the modeled vocabulary, or it was never revealed)."""
    if not value:
        return -1
    for i, group in enumerate(groups):
        if value in group:
            return i
    return -1


def _play_share(n, det, time_ms, idx, stamp, out_dir=None):
    import numpy as np
    from poke_env import AccountConfiguration
    from poke_env.data import to_id_str
    from laplace.agent.engine_search import EnginePlayer
    from laplace.policy.features import build_inputs, opponent_bench_species
    from laplace.policy.action_space import N_ACTIONS, targets_from_pooled
    from laplace.value.value_features import _ITEM_FLAGS, _ABILITY_FLAGS
    from laplace.policy.features import TERA_TYPES

    class DataPlayer(EnginePlayer):
        """EnginePlayer that records one (features, policy target) sample per decision, via
        the existing `observer` hook -- no changes to choose_move's control flow needed."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, observer=self._collect, **kwargs)
            self.samples = {}   # battle_tag -> list of (board, history, action_feats, mask,
                                # target, bench_species_snapshot)

        def _collect(self, event, **data):
            if event != "decision":
                return
            try:
                battle = data["battle"]
                ranked = data.get("ranked") or []
                if not ranked or not self._worlds:
                    return
                # First determinized world this turn -- see features.py's module docstring
                # on why action/board features don't need to vary per hidden world.
                state = self._worlds[0][0]
                hist = self._event_history.get(battle.battle_tag, [])
                inputs = build_inputs(state, battle, hist)
                target = targets_from_pooled(dict(ranked), inputs["slot_to_choice"])
                if sum(target) <= 0:
                    return   # nothing in `ranked` mapped to a legal slot this turn; skip
                self.samples.setdefault(battle.battle_tag, []).append((
                    inputs["board"].astype(np.float16),
                    inputs["history"].astype(np.float16),
                    inputs["action_feats"].astype(np.float16),
                    inputs["action_mask"],
                    np.asarray(target, dtype=np.float16),
                    opponent_bench_species(battle),   # resolved to final reveals post-game
                ))
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception:
                pass

    # Same rationale as gen_value_data.py: play with the LADDER root config (mixed strategy,
    # current nets) so the data distribution matches deployment, not engine defaults.
    _value_net = paths.VALUE_NET
    _policy_net = paths.POLICY_NET
    ladder_kw = dict(robust_vote=False, mix_root=True, mix_frac=0.9, value_boost_margin=0,
                     value_model_path=_value_net if os.path.exists(_value_net) else None,
                     policy_model_path=_policy_net if os.path.exists(_policy_net) else None)

    async def run():
        a = DataPlayer(account_configuration=AccountConfiguration.generate(f"pga{idx}", rand=True),
                       battle_format=BATTLE_FORMAT, n_determinizations=det,
                       search_time_ms=time_ms, threads=2, **ladder_kw)
        b = DataPlayer(account_configuration=AccountConfiguration.generate(f"pgb{idx}", rand=True),
                       battle_format=BATTLE_FORMAT, n_determinizations=det,
                       search_time_ms=time_ms, threads=2, **ladder_kw)
        await a.battle_against(b, n_battles=n)

        boards, hists, afeats, masks, targets, values, games = [], [], [], [], [], [], []
        item_targets, ability_targets, tera_targets = [], [], []
        game = 0
        for player in (a, b):
            for tag, samples in player.samples.items():
                battle = player.battles.get(tag)
                if battle is None or battle.won is None:   # drop ties / unfinished
                    continue
                label = 1.0 if battle.won else 0.0
                # Final, most-revealed state of the opponent's team, for hindsight labels --
                # strictly more informative than what was known at any earlier decision, by
                # construction of the game (reveals only accumulate).
                final_by_species = {to_id_str(m.species): m
                                    for m in battle.opponent_team.values()}
                for board, hist, afeat, mask, target, bench_species in samples:
                    boards.append(board)
                    hists.append(hist)
                    afeats.append(afeat)
                    masks.append(mask)
                    targets.append(target)
                    values.append(label)
                    games.append(game)

                    item_row = np.full(N_OPP_BENCH, -1, dtype=np.int8)
                    ability_row = np.full(N_OPP_BENCH, -1, dtype=np.int8)
                    tera_row = np.full(N_OPP_BENCH, -1, dtype=np.int8)
                    for slot, species in enumerate(bench_species):
                        mon = final_by_species.get(species) if species else None
                        if mon is None:
                            continue
                        item_row[slot] = _class_index(getattr(mon, "item", None), _ITEM_FLAGS)
                        ability_row[slot] = _class_index(getattr(mon, "ability", None),
                                                         _ABILITY_FLAGS)
                        tera = getattr(mon, "tera_type", None)
                        if tera is not None and getattr(mon, "is_terastallized", False):
                            tera_name = tera.name.lower()
                            tera_row[slot] = (TERA_TYPES.index(tera_name)
                                              if tera_name in TERA_TYPES else -1)
                    item_targets.append(item_row)
                    ability_targets.append(ability_row)
                    tera_targets.append(tera_row)
                game += 1
        if boards:
            return (np.stack(boards), np.stack(hists), np.stack(afeats), np.stack(masks),
                    np.stack(targets), np.asarray(values, np.float16),
                    np.asarray(games, np.int32), np.stack(item_targets),
                    np.stack(ability_targets), np.stack(tera_targets))
        from laplace.value.value_features import N_VALUE_FEATURES
        from laplace.policy.history import N_HISTORY, EVENT_DIM
        from laplace.policy.features import ACTION_FEATURE_DIM
        return (np.zeros((0, N_VALUE_FEATURES), np.float16),
                np.zeros((0, N_HISTORY, EVENT_DIM), np.float16),
                np.zeros((0, N_ACTIONS, ACTION_FEATURE_DIM), np.float16),
                np.zeros((0, N_ACTIONS), bool),
                np.zeros((0, N_ACTIONS), np.float16),
                np.zeros(0, np.float16), np.zeros(0, np.int32),
                np.zeros((0, N_OPP_BENCH), np.int8), np.zeros((0, N_OPP_BENCH), np.int8),
                np.zeros((0, N_OPP_BENCH), np.int8))

    (board, hist, afeat, mask, target, value, g, item_t, ability_t,
     tera_t) = asyncio.new_event_loop().run_until_complete(run())
    out_dir = out_dir or DATA_DIR
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"chunk_{stamp}_{idx}.npz")
    import numpy as np
    np.savez_compressed(path, board=board, hist=hist, afeat=afeat, mask=mask,
                        target=target, value=value, g=g, item_t=item_t,
                        ability_t=ability_t, tera_t=tera_t)
    return len(value), float(value.mean()) if len(value) else 0.0


def main():
    global DATA_DIR
    ap = argparse.ArgumentParser(description="Self-play data generation for the policy net.")
    ap.add_argument("--battles", type=int, default=200)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--det", type=int, default=4)
    ap.add_argument("--time-ms", type=int, default=60)
    ap.add_argument("--out-dir", default=None,
                    help="chunk output dir (default data_policy/)")
    args = ap.parse_args()
    if args.out_dir:
        DATA_DIR = os.path.abspath(args.out_dir)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    base, extra = divmod(args.battles, args.workers)
    shares = [base + (1 if i < extra else 0) for i in range(args.workers)]
    shares = [s for s in shares if s > 0]
    print(f"Generating from {args.battles} self-play games over {len(shares)} workers "
          f"(det={args.det}, {args.time_ms}ms) -> {DATA_DIR}", flush=True)

    t0 = time.time()
    total = 0
    with ProcessPoolExecutor(max_workers=len(shares)) as ex:
        futs = [ex.submit(_play_share, n, args.det, args.time_ms, i, stamp, DATA_DIR)
                for i, n in enumerate(shares)]
        for f in futs:
            n, mean = f.result()
            total += n
    dt = time.time() - t0
    print(f"Done: {total} samples from {args.battles} games in {dt:.0f}s "
          f"({total / max(dt, 1):.0f} samples/s).", flush=True)


if __name__ == "__main__":
    main()
