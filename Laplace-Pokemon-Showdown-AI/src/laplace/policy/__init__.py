r"""Learned policy prior for Laplace (Phase 1 of the Jaxcalibur-inspired upgrade).

Scope, honestly stated up front:

poke-engine's `monte_carlo_tree_search` (the pip-installed 0.0.47 build this project ships
with) ships as an opaque Rust function with no hook for an external prior. Rather than settle
for a weaker Python-side post-hoc reranking of the pooled result (this package's ORIGINAL,
more conservative plan), the accompanying poke-engine patch
(`poke-engine-root-prior.patch` / the patched `poke-engine-main/` in this delivery) adds a
real one: `side_one_prior`/`side_two_prior`/`root_c_puct`/`root_regret_matching` parameters
that inject this package's model into the ROOT of the actual Rust search as a PUCT or
regret-matching-flavoured prior -- see `mcts.rs`'s `RootPolicy` and the top-level README's
"What changed" section for the exact formulas and the one honestly-disclosed limitation
(it's root-only, and the two sides currently share one `RootPolicy`).

This package (`laplace/policy/`) is the Python side of that: feature extraction
(`features.py`, `history.py`, `action_space.py`), the network itself (`net.py`), and the
training pipeline (`../cli/gen_policy_data.py`, `../cli/train_policy.py`). `engine_search.py`
is where the two meet: `_policy_forward` runs the net once per turn and hands its outputs to
`monte_carlo_tree_search` (the action prior) and to `poke_engine_adapter.build_state`'s
`opp_posterior` (Phase 3's item/ability/Tera posterior, folded into the existing Random
Battle joint-sets sampler rather than replacing it).
"""
