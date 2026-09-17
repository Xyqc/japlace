r"""Phase 11: A/B benchmark two EnginePlayer configs against EACH OTHER (not a heuristic
punching bag), at matched search compute, and report what's actually measurable without a
live ladder connection.

What this measures, and what it can't:
    CAN measure locally: win rate (challenger vs baseline), average decision latency,
    resignation/fallback/error counts (from each player's `.diag`), and -- since both
    players are `EnginePlayer` -- this is an apples-to-apples comparison at whatever
    determinizations/search-time/threads you pass both sides.
    CANNOT measure here: Elo, GXE, or anything that needs the real ladder's player pool.
    Win rate against a fixed baseline is the right proxy for "did this change help", but
    it is not the same number as ladder Elo, and shouldn't be reported as one.

Two configs are built from the SAME base kwargs plus an override dict each, so a config
like `{"policy_model_path": "models/policy_net.pt"}` isolates exactly the thing being
tested. Example configs to compare, none of which are assumed to help without running this:
    root prior on/off:        {} vs {"policy_model_path": "models/policy_net.pt"}
    regret-matching vs PUCT:  {"policy_model_path": ..., "policy_weight": 0.35}
                              vs the same + a regret-matching flag once one exists on
                              EnginePlayer (today root_regret_matching is only reachable by
                              editing engine_search.py's monte_carlo_tree_search call sites
                              directly -- see the README's Phase 6 section)
    opponent heuristic on/off: {} vs {"use_opponent_prior": True}

From the project root, with the local server running:
    python -m laplace.cli.bench_ab --battles 100 --workers 10
    python -m laplace.cli.bench_ab --challenger-kwargs '{"use_opponent_prior": true}'
"""

import argparse
import asyncio
import json
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

BATTLE_FORMAT = "gen9randombattle"

BASE_KWARGS = dict(n_determinizations=8, search_time_ms=450, threads=2,
                   robust_vote=False, mix_root=True, mix_frac=0.9)


def _play_share(n, base_kwargs, baseline_kwargs, challenger_kwargs, idx):
    """Worker process: play `n` A-vs-B battles, return summary stats. Top-level so it
    pickles on Windows (see eval_engine.py's same note)."""
    from poke_env import AccountConfiguration
    from laplace.agent.engine_search import EnginePlayer

    async def run():
        baseline = EnginePlayer(
            account_configuration=AccountConfiguration.generate(f"bab{idx}", rand=True),
            battle_format=BATTLE_FORMAT, record=False,
            **{**base_kwargs, **baseline_kwargs})
        challenger = EnginePlayer(
            account_configuration=AccountConfiguration.generate(f"cab{idx}", rand=True),
            battle_format=BATTLE_FORMAT, record=False,
            **{**base_kwargs, **challenger_kwargs})

        latencies = {"baseline": [], "challenger": []}
        orig_choose = {}
        for name, player in (("baseline", baseline), ("challenger", challenger)):
            orig = player.choose_move
            def wrapped(battle, _orig=orig, _name=name):
                t0 = time.perf_counter()
                order = _orig(battle)
                latencies[_name].append((time.perf_counter() - t0) * 1000)
                return order
            player.choose_move = wrapped

        await challenger.battle_against(baseline, n_battles=n)
        return {
            "challenger_wins": challenger.n_won_battles,
            "baseline_wins": baseline.n_won_battles,
            "n": n,
            "challenger_diag": dict(challenger.diag),
            "baseline_diag": dict(baseline.diag),
            "challenger_latency_ms": latencies["challenger"],
            "baseline_latency_ms": latencies["baseline"],
        }

    return asyncio.new_event_loop().run_until_complete(run())


def _split(total, workers):
    base, extra = divmod(total, workers)
    return [base + (1 if i < extra else 0) for i in range(workers)
            if base + (1 if i < extra else 0) > 0]


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser(description="A/B benchmark two EnginePlayer configs.")
    ap.add_argument("--battles", type=int, default=100)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--determinizations", type=int, default=8)
    ap.add_argument("--time-ms", type=int, default=450)
    ap.add_argument("--threads", type=int, default=2,
                    help="per-battle search threads; kept low here since --workers already "
                         "parallelizes across battles -- see eval_engine.py's note on the "
                         "two knobs")
    ap.add_argument("--baseline-kwargs", type=str, default="{}",
                    help="JSON dict of EnginePlayer kwargs for the baseline (control)")
    ap.add_argument("--challenger-kwargs", type=str, default='{"policy_model_path": "models/policy_net.pt"}',
                    help="JSON dict of EnginePlayer kwargs for the challenger (treatment)")
    args = ap.parse_args()

    base_kwargs = dict(BASE_KWARGS, n_determinizations=args.determinizations,
                       search_time_ms=args.time_ms, threads=args.threads)
    baseline_kwargs = json.loads(args.baseline_kwargs)
    challenger_kwargs = json.loads(args.challenger_kwargs)

    shares = _split(args.battles, args.workers)
    print(f"A/B: challenger {challenger_kwargs} vs baseline {baseline_kwargs}, "
          f"det={args.determinizations} {args.time_ms}ms threads={args.threads}, "
          f"{args.battles} battles over {len(shares)} workers.", flush=True)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=len(shares)) as ex:
        futures = [ex.submit(_play_share, n, base_kwargs, baseline_kwargs, challenger_kwargs, i)
                  for i, n in enumerate(shares)]
        results = [f.result() for f in futures]
    dt = time.time() - t0

    total = sum(r["n"] for r in results)
    challenger_wins = sum(r["challenger_wins"] for r in results)
    baseline_wins = sum(r["baseline_wins"] for r in results)
    unresolved = total - challenger_wins - baseline_wins   # crashes / disconnects / ties

    challenger_diag, baseline_diag = Counter(), Counter()
    challenger_lat, baseline_lat = [], []
    for r in results:
        challenger_diag.update(r["challenger_diag"])
        baseline_diag.update(r["baseline_diag"])
        challenger_lat += r["challenger_latency_ms"]
        baseline_lat += r["baseline_latency_ms"]

    wr = challenger_wins / max(challenger_wins + baseline_wins, 1)
    print(f"\nChallenger win rate: {wr:.1%}  ({challenger_wins}-{baseline_wins}, "
          f"{unresolved} unresolved)   [{dt:.0f}s wall for {total} battles]")
    print(f"Decision latency (ms):  challenger mean={_mean(challenger_lat):.0f}  "
          f"baseline mean={_mean(baseline_lat):.0f}")
    print(f"Challenger diag: {dict(challenger_diag)}")
    print(f"Baseline diag:   {dict(baseline_diag)}")
    print("\nNote: this is win rate at matched local self-play, not ladder Elo/GXE -- see "
          "this file's module docstring.")


if __name__ == "__main__":
    main()
