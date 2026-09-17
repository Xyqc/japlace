r"""Train the Phase-1 policy net: distill the search+guards' final decision, plus a value
head trained the same way train_value.py's is (P(win) from the board token).

Data: data_policy/chunk_*.npz from gen_policy_data.py -- board/hist/afeat/mask/target/value
float arrays, g per-sample game index within the chunk. Same by-GAME train/val split as
train_value.py, for the same reason (samples within one game are correlated).

Losses, and why each is here (Phase 8's brief: "explain why each hyperparameter is
appropriate" rather than copying Jaxcalibur's numbers blind):

  * policy: soft cross-entropy against the search+guards' target distribution (not hard
    argmax) -- the target already IS a distribution (visit shares run through the guard
    pipeline), so training against it directly preserves the information about how close the
    top actions were, which a hard label would throw away.
  * value: MSE against the game outcome mapped to [-1, 1] (matching the net's tanh head) --
    plain MSE rather than BCE-with-logits because the head's activation is already a bounded
    tanh, not a sigmoid; squashing through log-odds a second time buys nothing here.
  * entropy bonus (subtracted from the loss, i.e. added as a bonus): keeps the policy from
    collapsing onto the teacher's occasional near-one-hot turns (e.g. a forced move with only
    one legal option) in a way that also flattens turns with genuinely close options. Small
    by design (`--entropy-coef`, default 0.01) -- this is a regularizer, not the training
    objective.
  * zero-avoiding regularization: implemented as label smoothing on the target distribution
    (mix in a small uniform component over LEGAL actions only, before computing the
    cross-entropy) rather than a separate loss term. Jaxcalibur's write-up names this
    technique but doesn't give its exact form; label smoothing is the standard, well-
    understood instance of "never let the target model assign exactly zero to a legal
    action", and composes cleanly with soft-label CE above instead of needing its own
    weighting term to tune.
  * item/ability/Tera auxiliary losses (Phase 3): plain cross-entropy per bench slot, masked
    to slots gen_policy_data.py could actually resolve a hindsight label for (a slot's
    target is -1, meaning "never revealed this game", far more often than not -- see that
    file's docstring). Weighted low (`--aux-weight`, default 0.3) relative to the policy/
    value losses: these heads are a secondary, sparser signal riding on the same forward
    pass, not the primary training objective, and shouldn't be allowed to dominate the
    shared encoder's gradient just because there are three of them.

    python -m laplace.cli.train_policy                  # -> models/policy_net.pt
    python -m laplace.cli.train_policy --epochs 30 --entropy-coef 0.02
"""

import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from laplace import paths
from laplace.policy.net import PolicyNet, count_parameters

DATA_GLOB = os.path.join(paths.POLICY_DATA_DIR, "chunk_*.npz")
MODEL_PATH = paths.POLICY_NET


def load_split(val_frac=0.1, seed=7):
    boards, hists, afeats, masks, targets, values, gids = [], [], [], [], [], [], []
    item_ts, ability_ts, tera_ts = [], [], []
    gid = 0
    for path in sorted(glob.glob(DATA_GLOB)):
        d = np.load(path)
        if not len(d["value"]):
            continue
        boards.append(d["board"].astype(np.float32))
        hists.append(d["hist"].astype(np.float32))
        afeats.append(d["afeat"].astype(np.float32))
        masks.append(d["mask"].astype(bool))
        targets.append(d["target"].astype(np.float32))
        values.append(d["value"].astype(np.float32))
        g = d["g"].astype(np.int64)
        gids.append(g + gid)
        gid += int(g.max()) + 1 if len(g) else 0
        # Older chunks (generated before Phase 3's aux targets existed) won't have these
        # keys -- default to "no label anywhere" rather than erroring, so a mixed set of
        # old/new chunks still trains (just with less aux signal from the old ones).
        n = len(g)
        item_ts.append(d["item_t"].astype(np.int64) if "item_t" in d.files
                       else np.full((n, 6), -1, dtype=np.int64))
        ability_ts.append(d["ability_t"].astype(np.int64) if "ability_t" in d.files
                          else np.full((n, 6), -1, dtype=np.int64))
        tera_ts.append(d["tera_t"].astype(np.int64) if "tera_t" in d.files
                       else np.full((n, 6), -1, dtype=np.int64))
    if not boards:
        raise SystemExit(f"no data found under {DATA_GLOB} -- run gen_policy_data.py first")

    board = np.concatenate(boards)
    hist = np.concatenate(hists)
    afeat = np.concatenate(afeats)
    mask = np.concatenate(masks)
    target = np.concatenate(targets)
    value = np.concatenate(values)
    g = np.concatenate(gids)
    item_t = np.concatenate(item_ts)
    ability_t = np.concatenate(ability_ts)
    tera_t = np.concatenate(tera_ts)

    rng = np.random.default_rng(seed)
    unique_games = np.unique(g)
    rng.shuffle(unique_games)
    n_val_games = max(1, int(len(unique_games) * val_frac))
    val_games = set(unique_games[:n_val_games].tolist())
    val_mask = np.array([gi in val_games for gi in g])

    def pick(m):
        return (board[m], hist[m], afeat[m], mask[m], target[m], value[m],
                item_t[m], ability_t[m], tera_t[m])

    return pick(~val_mask), pick(val_mask)


def smooth_targets(target, mask, smoothing):
    """Zero-avoiding regularization: blend `smoothing` of a uniform distribution over this
    turn's LEGAL actions into the target, per-sample. `target`/`mask`: [B, N_ACTIONS]."""
    legal_count = mask.sum(dim=1, keepdim=True).clamp(min=1)
    uniform = mask.float() / legal_count
    return (1 - smoothing) * target + smoothing * uniform


def masked_aux_loss(logits, target_idx):
    """logits: [B, 6, C], target_idx: [B, 6] with -1 meaning "no label" (see
    gen_policy_data.py). Cross-entropy over labelled slots only; returns (loss, n_labelled,
    n_correct) so the caller can report accuracy and skip the loss term entirely on a batch
    with no labels at all (very possible for tera specifically -- see that file's docstring
    on how rarely it gets revealed) without dividing by zero."""
    valid = target_idx >= 0
    n_valid = int(valid.sum().item())
    if n_valid == 0:
        return logits.new_zeros(()), 0, 0
    flat_logits = logits[valid]                       # [n_valid, C]
    flat_target = target_idx[valid]                    # [n_valid]
    loss = F.cross_entropy(flat_logits, flat_target)
    correct = int((flat_logits.argmax(dim=-1) == flat_target).sum().item())
    return loss, n_valid, correct


def run_epoch(model, data, optimizer, entropy_coef, smoothing, aux_weight, batch_size,
             device, train):
    board, hist, afeat, mask, target, value, item_t, ability_t, tera_t = data
    n = len(value)
    idx = np.random.permutation(n) if train else np.arange(n)
    model.train(train)

    total_loss = total_policy = total_value = total_entropy = 0.0
    total_item = total_ability = total_tera = 0.0
    top1_correct = 0
    item_correct = item_labelled = ability_correct = ability_labelled = 0
    tera_correct = tera_labelled = 0
    n_batches = 0

    for start in range(0, n, batch_size):
        bidx = idx[start:start + batch_size]
        if len(bidx) == 0:
            continue
        b_board = torch.from_numpy(board[bidx]).to(device)
        b_hist = torch.from_numpy(hist[bidx]).to(device)
        b_afeat = torch.from_numpy(afeat[bidx]).to(device)
        b_mask = torch.from_numpy(mask[bidx]).to(device)
        b_target = torch.from_numpy(target[bidx]).to(device)
        b_value = torch.from_numpy(value[bidx]).to(device) * 2.0 - 1.0   # {0,1} -> [-1,1]
        b_item_t = torch.from_numpy(item_t[bidx]).to(device)
        b_ability_t = torch.from_numpy(ability_t[bidx]).to(device)
        b_tera_t = torch.from_numpy(tera_t[bidx]).to(device)

        with torch.set_grad_enabled(train):
            out = model(b_board, b_hist, b_afeat, b_mask)
            log_probs = F.log_softmax(out["policy_logits"], dim=-1)
            # Illegal slots carry log_probs = -inf (see PolicyNet.forward's masked_fill).
            # smoothed/target are exactly 0 there, but 0 * -inf = NaN in IEEE float, not 0 --
            # torch.where masks those terms out explicitly rather than relying on the
            # multiplication to zero them.
            zeros = torch.zeros_like(log_probs)
            smoothed = smooth_targets(b_target, b_mask, smoothing)
            policy_terms = torch.where(b_mask, smoothed * log_probs, zeros)
            policy_loss = -policy_terms.sum(dim=-1).mean()

            probs = log_probs.exp()
            entropy_terms = torch.where(b_mask, probs * log_probs, zeros)
            entropy = -entropy_terms.sum(dim=-1).mean()

            value_loss = F.mse_loss(out["value"], b_value)

            item_loss, n_item, c_item = masked_aux_loss(out["item_logits"], b_item_t)
            ability_loss, n_ability, c_ability = masked_aux_loss(out["ability_logits"],
                                                                   b_ability_t)
            tera_loss, n_tera, c_tera = masked_aux_loss(out["tera_logits"], b_tera_t)
            aux_loss = item_loss + ability_loss + tera_loss

            loss = policy_loss + value_loss - entropy_coef * entropy + aux_weight * aux_loss

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * len(bidx)
        total_policy += policy_loss.item() * len(bidx)
        total_value += value_loss.item() * len(bidx)
        total_entropy += entropy.item() * len(bidx)
        total_item += item_loss.item() * len(bidx)
        total_ability += ability_loss.item() * len(bidx)
        total_tera += tera_loss.item() * len(bidx)
        item_correct += c_item; item_labelled += n_item
        ability_correct += c_ability; ability_labelled += n_ability
        tera_correct += c_tera; tera_labelled += n_tera
        top1_correct += (out["policy_logits"].argmax(dim=-1)
                          == b_target.argmax(dim=-1)).sum().item()
        n_batches += len(bidx)

    n_batches = max(n_batches, 1)
    return {
        "loss": total_loss / n_batches,
        "policy_loss": total_policy / n_batches,
        "value_loss": total_value / n_batches,
        "entropy": total_entropy / n_batches,
        "item_loss": total_item / n_batches,
        "ability_loss": total_ability / n_batches,
        "tera_loss": total_tera / n_batches,
        "top1_acc": top1_correct / n_batches,
        "item_acc": item_correct / max(item_labelled, 1),
        "ability_acc": ability_correct / max(ability_labelled, 1),
        "tera_acc": tera_correct / max(tera_labelled, 1),
        "item_labelled": item_labelled,
        "ability_labelled": ability_labelled,
        "tera_labelled": tera_labelled,
    }


def main():
    ap = argparse.ArgumentParser(description="Train the Phase-1 policy net.")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--entropy-coef", type=float, default=0.01)
    ap.add_argument("--smoothing", type=float, default=0.03,
                    help="zero-avoiding regularization strength (label smoothing over legal actions)")
    ap.add_argument("--aux-weight", type=float, default=0.3,
                    help="weight on the summed item+ability+Tera auxiliary losses (Phase 3)")
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_data, val_data = load_split(seed=args.seed)
    print(f"train: {len(train_data[-1])} samples, val: {len(val_data[-1])} samples", flush=True)

    device = "cpu"   # matches the project's CPU-inference constraint; training is cheap at
                      # this model size, and keeping train/infer on the same backend avoids
                      # a class of "works in training, panics at inference" surprises.
    model = PolicyNet(d_model=args.d_model, n_layers=args.n_layers).to(device)
    print(f"model parameters: {count_parameters(model):,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_loss = float("inf")
    best_state = None
    epochs_since_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_data, optimizer, args.entropy_coef,
                                  args.smoothing, args.aux_weight, args.batch_size, device,
                                  train=True)
        val_metrics = run_epoch(model, val_data, optimizer, args.entropy_coef,
                                args.smoothing, args.aux_weight, args.batch_size, device,
                                train=False)
        print(f"epoch {epoch:3d} | train loss {train_metrics['loss']:.4f} "
              f"(pol {train_metrics['policy_loss']:.4f} val {train_metrics['value_loss']:.4f} "
              f"ent {train_metrics['entropy']:.3f} top1 {train_metrics['top1_acc']:.3f}) | "
              f"val loss {val_metrics['loss']:.4f} top1 {val_metrics['top1_acc']:.3f} | "
              f"aux acc item {val_metrics['item_acc']:.2f}(n={val_metrics['item_labelled']}) "
              f"ability {val_metrics['ability_acc']:.2f}(n={val_metrics['ability_labelled']}) "
              f"tera {val_metrics['tera_acc']:.2f}(n={val_metrics['tera_labelled']})",
              flush=True)

        if val_metrics["loss"] < best_val_loss - 1e-4:
            best_val_loss = val_metrics["loss"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= args.patience:
                print(f"early stopping at epoch {epoch} (no val improvement for "
                      f"{args.patience} epochs)", flush=True)
                break

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save({
        "state_dict": best_state or model.state_dict(),
        "arch_kwargs": {"d_model": args.d_model, "n_layers": args.n_layers},
        "best_val_loss": best_val_loss,
    }, MODEL_PATH)
    print(f"saved -> {MODEL_PATH}", flush=True)


if __name__ == "__main__":
    main()
