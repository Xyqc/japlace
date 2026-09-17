# Laplace x Jaxcalibur upgrade: final delivery and honest status

Two repos, patches already applied, ready to use. This README replaces the earlier one and
covers everything across both rounds of work.

```
poke-engine-main/                  <- patched poke-engine (Rust engine + Python bindings)
Laplace-Pokemon-Showdown-AI/       <- patched Laplace (server/ removed -- see below)
poke-engine-root-prior.patch
laplace-phase1-policy-net.patch
```

`Laplace-Pokemon-Showdown-AI/server/` (the Pokemon Showdown server checkout) was removed --
it's a 130+MB third-party clone untouched by any of this work, and your existing local copy
(or the project README's `git clone`/`npm install` step) already has it. Also removed:
`.git/`, `replays/` (runtime output).

## Read this first: what "done" means here

I do not have a live Pokemon Showdown connection, your ladder account, or a Rust toolchain
newer than 1.75 in this environment (the real one, `pyo3 0.29`, needs >=1.83; the servers
that would give me a newer one aren't reachable from here). That means, permanently, for
this entire engagement:

- **No Elo, no GXE, no ladder win rate.** Nothing in this delivery is a claim that the bot
  is stronger. Every number in this README is either a unit test result, a synthetic-data
  training smoke test, or a measured (not estimated) CPU timing -- never a game-outcome
  metric, because I have never played a real game with any of this code.
- **The PyO3 binding layer (`poke-engine-py`) was never compiled**, only hand-reviewed
  against the existing file's own macro conventions. Run `cargo check` there yourself
  before trusting it.

Everything else below -- the architecture, the Rust patch, the training pipeline, the
posterior wiring -- is real code that compiles, imports, and does what its tests show it
doing. "All phases" is complete in the sense of "every phase has been designed, implemented
as far as it can be without a live connection, and honestly scoped where it can't be." It is
not complete in the sense of "proven to make the ladder bot stronger" -- that step is yours
to run, with the commands below.

## Phase-by-phase status

| Phase | What the brief asked for | Status |
|---|---|---|
| 1. Policy network | Learned prior feeding search | **Done + verified.** `PolicyNet` (404,010 params), fed into poke-engine's actual root MCTS via a real Rust patch (`side_one_prior`/`root_c_puct`), not just a post-hoc rerank. |
| 2. History/temporal repr | Recent-event representation | **Partially done.** 16-event window from poke-env's own protocol log, additive to the existing trackers. It's a hand-encoded fixed-width feature (side/kind/tera/recency), not learned embeddings over move/species identity -- a real simplification, not the richer representation implied. |
| 3. Item/ability/Tera prediction | Aux heads conditioning the hidden-world sampler | **Done + verified.** Heads exist, have real hindsight-labelled training targets, and their output is wired into `poke_engine_adapter._JointSets.sample` as a multiplicative reweight of the existing Random-Battle-stats prior -- "prior + revealed-info filtering + neural posterior", per the brief, not neural-alone. Unit-tested end to end (see below). |
| 4. Opponent modelling | Learned opponent-action prior | **Not done as specified.** `use_opponent_prior` exists but is a hand-tuned heuristic (revealed moves scored by `estimate_damage_fraction`), not a learned model -- `net.py`'s `opp_action_head` exists but has no established correspondence between its 16 generic slots and a specific opponent's actual moves. Fixing that needs redesigning the head around opponent-specific action features, which is real, separate work not attempted here. |
| 5. Prevent hidden-info leakage | Opponent search can't see our sampled world | **Done, structurally.** The opp_action head only ever reads board+history (documented, by construction). `side_two_prior` is a real Rust parameter now, with an explicit, repeated warning in three places (poke-engine's own docstring, the Python wrapper, and this README) that the engine cannot verify what a caller puts in it -- the discipline is entirely in `engine_search.py` only ever filling it from revealed information. |
| 6. Root policy / regret-matching | pUCT/regret-matching investigated and benchmarked | **Implemented, NOT benchmarked.** Both PUCT and regret-matching selection exist in the Rust patch (`RootPolicy`). `bench_ab.py` (new) is the harness to A/B them locally. I did not run it -- no working `poke_engine` build in this environment, see above. |
| 7. Richer state encoder | Transformer-style, reuse existing features | **Partially done.** The transformer exists and reuses the 368-dim value-net feature vector wholesale for its board token, exactly as asked. It does NOT add richer per-Pokemon tokens (own team, opponent team) beyond that single flat vector -- a real capacity gap flagged, not silently accepted. |
| 8. Training pipeline | Dataset gen, training, validation, checkpointing | **Done + verified.** `gen_policy_data.py` (self-play via the existing `observer` hook, now including item/ability/Tera hindsight labels) and `train_policy.py` (policy CE, value MSE, entropy bonus, label-smoothing "zero-avoiding" reg, masked aux losses). Ran end-to-end on synthetic data; found and fixed a real NaN-gradient bug before it could hit real data. |
| 9. Preserve guards | Keep + evaluate for redundancy | **Analysis only, no code change** (see below) -- I don't have the A/B infrastructure running to prove redundancy either way, so nothing was removed. |
| 10. Resignation | Keep conservative | **Untouched code, one real interaction flagged** (see below). |
| 11. Benchmarking | Compare old vs new under matched compute | **Harness built, not run.** `bench_ab.py` measures win rate, decision latency, and diag counters (resignations, fallbacks, errors) for two configs head-to-head. No Elo/GXE (needs the ladder). No numbers reported, because none were produced. |
| 12. Human-like timing | Keep ~450ms/world | **Verified compatible.** Measured (not estimated): the policy net's forward pass is ~1ms single-threaded CPU -- negligible against the 450ms/world budget. Root prior injection doesn't add search time; it changes which options get visited within the same time budget. |
| 13. Code quality / Windows | No POSIX-only code, fail safely | **Reviewed, no issues found** in the new code (see below); nothing here was tested ON Windows, only reviewed for obviously non-portable patterns. |

## Phase 9: guard redundancy analysis (no code changed)

The guard pipeline (`absorb` -> `futility` -> `tiebreak` -> `value` -> `gamble` -> `noop` ->
`deadlock`, see `choose_move`'s `_stage` calls) runs entirely AFTER search, on the pooled
result. The root prior (Phases 1/3/4) operates INSIDE search, shifting which options get
visited -- a different layer of the pipeline entirely. None of the five guards became
redundant by inspection: each encodes a DETERMINISTIC fact (an immunity, a Choice lock, a
true no-op) that no amount of statistical prior-shifting inside search changes or duplicates.
The `value` stage (the existing value-net rerank) is the closest conceptual neighbor to the
new policy prior -- both are learned, both are statistical -- but they score different
things (board evaluation vs. action preference) from different features, so "redundant" isn't
the right frame there either. This is reasoned analysis, not a benchmark result; Phase 9
explicitly wants that benchmark before concluding anything, which I can't run here.

## Phase 10: resignation interaction (one real thing to know)

`_should_resign` itself is unmodified. But it reads `pooled` -- the raw per-world MCTS visit
shares -- and `pooled` DOES shift when a root prior is active, because PUCT/regret-matching
change where visits concentrate within a world's search. This doesn't touch resignation's
actual safety property (it still requires multi-world agreement plus no credible recovery
line, "uncertainty means CONTINUE"), but if you A/B a root-prior config and see a different
resignation RATE, that's an expected, real interaction to account for, not a bug in either
system.

## What's new since the previous delivery (Phase 3 + benchmarking harness)

- `poke_engine_adapter.py`: `_JointSets.sample(..., posterior=None)` and
  `_posterior_multiplier` -- multiplies observed-count weights by a neural
  item/ability/Tera posterior when supplied, neutral (1.0x) otherwise. Threaded through
  `_opp_pokemon_determinized` -> `_opp_side` -> `build_state(..., opp_posterior=...)`.
- `engine_search.py`: `_policy_prior_dict` replaced by `_policy_forward`, which runs the net
  ONCE per turn and returns both the action prior AND the hidden posterior (previously would
  have needed two forward passes for two features -- now one).
- `features.py`: `opponent_bench_species` (canonical slot order, MUST match
  `poke_engine_adapter._opp_side`'s own ordering -- documented in both places),
  `expand_group_posterior`, `TERA_TYPES`.
- `gen_policy_data.py` / `train_policy.py`: real hindsight-labelled item/ability/Tera
  training targets and masked auxiliary losses (`--aux-weight`, default 0.3).
- `bench_ab.py` (new): local A/B harness, two `EnginePlayer` configs head-to-head, matched
  search compute, win rate + latency + diag counters. Not run.
- `policy/__init__.py`: corrected -- the original version of this file (from the first
  delivery) described a Python-side post-hoc reranking plan that was superseded once the
  real Rust patch landed. Fixed to describe what's actually implemented.

### Verified by actually running (this round)

- `_JointSets.sample`'s posterior reweighting: no posterior -> ~uniform draws over equal-
  count candidates (measured ~1/3 each over 3000 draws); a posterior favoring one candidate
  90% -> that candidate drawn ~90% of the time. The mechanism does what it's supposed to.
- `PolicyNet`'s item/ability/tera head outputs feed correctly into `expand_group_posterior`
  (valid probability distributions in, sensible per-id dicts out).
- `train_policy.py` end-to-end on synthetic data including the new aux targets: no NaNs,
  and per-head validation accuracy on RANDOM synthetic labels landed at chance level for
  each head (item ~1/6, ability ~1/9, tera ~1/18) -- confirming the masking/indexing is
  correct, not just "doesn't crash".
- `_class_index` (item/ability id -> group index for training labels) against known ids.
- Policy net forward-pass latency: ~1ms/call, single-threaded CPU, measured over 200 calls.
- Every touched/new Python file compiles and imports cleanly (`py_compile`, plus real
  imports against a stub `poke_engine` module for anything gated on the native extension).
- The poke-engine Rust patch still compiles clean (`cargo check --all-targets --features
  gen9,terastallization`) after this round's changes (which were Python-only, but
  re-verified as a full regression check, not assumed unaffected).

## Commands

```bash
# Rebuild poke-engine (your existing build process -- README's maturin/cargo steps)
cd poke-engine-main/poke-engine-py && maturin develop --release --features poke-engine/gen9,poke-engine/terastallization

# Generate self-play data (now includes item/ability/Tera hindsight labels)
cd ../../Laplace-Pokemon-Showdown-AI
python -m laplace.cli.gen_policy_data --battles 500 --workers 10 --det 4 --time-ms 60

# Train (policy + value + item/ability/Tera aux losses)
python -m laplace.cli.train_policy --epochs 30

# A/B benchmark a config against baseline (local self-play, no ladder needed)
python -m laplace.cli.bench_ab --battles 100 --workers 10 \
    --challenger-kwargs '{"policy_model_path": "models/policy_net.pt"}'

# Ladder with the trained policy net (picked up automatically once models/policy_net.pt exists)
python -m laplace.cli.ladder --format gen9randombattle
```

To try the opponent heuristic, pass `use_opponent_prior=True` in `build_agent`'s kwargs in
`ladder.py` (not yet its own CLI flag, since it hasn't been A/B'd).

## What I could and couldn't verify (carried forward + this round)

Could NOT verify, same as before, and still true:
- `poke-engine-py`'s PyO3 layer -- needs rustc >=1.83, unavailable here.
- Anything needing a live Showdown connection or the compiled `poke_engine` module: actual
  self-play games, ladder strength, Elo, GXE, whether the opponent heuristic or the Phase 3
  posterior help or hurt in practice.

## What's still genuinely NOT done, no hedging

- Phase 4's real learned opponent-action model (heuristic only).
- Phase 2's richer temporal representation (basic fixed-width encoding only).
- Phase 7's richer per-Pokemon token encoder (still one flat board vector).
- Per-node (not just root) prior injection -- would need NN inference embedded inside
  poke-engine's Rust loop, a substantially larger undertaking.
- Splitting `RootPolicy` so the two sides can be independently on/off (documented coupling,
  not fixed).
- ANY benchmark numbers. Zero. The harness exists; it has not been run.
- Windows testing (only static review).

## If you want to keep going from here

In order of expected payoff, given what's now built:
1. Get `poke-engine-py` compiling and built on your machine (real rustc), then actually run
   `bench_ab.py` -- this is the single highest-value next step, since literally nothing
   claimed here has been checked against a real game yet.
2. Generate a real self-play dataset (hundreds to low thousands of games) and train an
   actual checkpoint; check its policy top-1 agreement and item/ability/Tera accuracy
   against real (not synthetic-random) data before trusting it.
3. A/B `root_c_puct` on/off, then PUCT vs regret-matching, then the opponent heuristic
   on/off, each in isolation, with `bench_ab.py`.
4. Only after (1)-(3) show something real: invest in the bigger remaining items (per-node
   prior, a real opponent-action model, richer encoders).
