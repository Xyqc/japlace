# japlace: Modified Pokémon Showdown AI

A heavily modified version of **japlace**, using a patched **poke-engine** backend with a learned policy network and additional search improvements for Pokémon Showdown.

The project combines Laplace's existing battle AI with:

* Learned policy priors for MCTS
* Item, Ability, and Tera prediction
* Hidden-information posterior sampling
* Battle history features
* PUCT / regret-matching root search
* Optional opponent-action heuristics
* Policy training and self-play data generation
* Built-in A/B benchmarking

## Setup

### Build poke-engine

```bash
cd poke-engine-main/poke-engine/poke-engine-py
maturin develop --release --features poke-engine/gen9,poke-engine/terastallization
```

### Enter the AI project

```bash
cd ../../Laplace-Pokemon-Showdown-AI
```

Activate your virtual environment:

```cmd
.venv\Scripts\activate
```

## Generate Training Data

```bash
python -m laplace.cli.gen_policy_data --battles 500 --workers 10 --det 4 --time-ms 60
```

Training data is written to:

```text
data_policy/
```

## Train the Policy Network

```bash
python -m laplace.cli.train_policy --epochs 30
```

The trained model is saved as:

```text
models/policy_net.pt
```

## Benchmark

Compare the trained policy against the baseline:

```bash
python -m laplace.cli.bench_ab --battles 100 --workers 10 --challenger-kwargs "{\"policy_model_path\":\"models/policy_net.pt\"}"
```

## Play on Pokémon Showdown

Once `models/policy_net.pt` exists:

```bash
python -m laplace.cli.ladder --format gen9randombattle
```

The ladder client automatically loads the trained policy model.

## Project Structure

```text
japlace/
├── poke-engine-main/
│   └── poke-engine/
├── Laplace-Pokemon-Showdown-AI/
│   ├── laplace/
│   ├── models/
│   └── data_policy/
├── poke-engine-root-prior.patch
└── laplace-phase1-policy-net.patch
```

This repository is a custom **japlace modification**, combining the original Laplace battle system with new neural-policy, search, and hidden-information features.
