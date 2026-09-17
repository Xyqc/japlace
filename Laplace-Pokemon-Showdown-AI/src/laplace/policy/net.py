r"""PolicyNet: a small transformer over (board, history, action) tokens.

Architecture is inspired by Jaxcalibur's write-up (https://jaxcalibur.github.io/,
"Architecture" section) but is our own implementation -- Jaxcalibur's code and weights were
not released (as of the write-up's "Parting Thoughts", explicitly to limit how easily its
strength could be reproduced against the live human ladder), so nothing here is copied from
it, only the ideas it describes in prose:

    input tokens (board / active / moves / team / opponent team / history / matchups)
        -> transformer -> action logits + value + opponent-action + item/ability/Tera heads

Simplifications made for this phase, stated plainly rather than silently:

  * Jaxcalibur uses ~35 tokens with a position-based mixture-of-experts (different MLP
    weights per token TYPE). This implementation uses 3 token types (board/history/action)
    with per-type input projections into a shared transformer body -- the same idea at much
    lower parameter cost, appropriate for the ~1-3M-parameter / CPU-inference budget this
    project asked for (Jaxcalibur is 8.5M and targets GPU inference).
  * Jaxcalibur's 24 matchup tokens (per-pair damage calcs) are folded into the per-action
    "effectiveness" scalar in `features.action_features` instead of their own token stream.
    This is a real capacity reduction; revisit if policy-vs-search agreement plateaus below
    the value net's own accuracy.
  * The opponent-action head reuses the 16-slot action layout but is scored from
    board+history only (we cannot construct the opponent's own action-feature tokens without
    already knowing their hidden set, which is exactly the information the model must not
    assume). This is the Phase-5 leakage boundary made structural rather than just
    documented: the opponent-action head architecturally cannot see our sampled hidden-world
    features, only revealed board state and history.
  * Item/ability/Tera heads (Phase 3) are included but only as classifiers over the
    opponent's revealed bench slots using the existing flag vocabularies from
    `value.value_features` (`_ITEM_FLAGS`, `_ABILITY_FLAGS`) and the 18 Pokemon types for
    Tera -- deliberately reusing the same groupings the value net and the deterministic
    knowledge base already agree on, rather than inventing a second vocabulary to keep in
    sync. They are NOT yet wired into the hidden-world sampler (that's Phase 4/5 work on top
    of this checkpoint, not done in this pass).
"""

import torch
import torch.nn as nn

from laplace.value.value_features import N_VALUE_FEATURES, _ITEM_FLAGS, _ABILITY_FLAGS
from laplace.policy.action_space import N_ACTIONS
from laplace.policy.history import N_HISTORY, EVENT_DIM
from laplace.policy.features import ACTION_FEATURE_DIM

N_ITEM_CLASSES = len(_ITEM_FLAGS) + 1     # +1 "other/unknown"
N_ABILITY_CLASSES = len(_ABILITY_FLAGS) + 1
N_TERA_CLASSES = 18                       # 18 Pokemon types
N_OPP_BENCH = 6


class PolicyNet(nn.Module):
    def __init__(self, d_model=96, n_layers=3, n_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        self.board_proj = nn.Linear(N_VALUE_FEATURES, d_model)
        self.history_proj = nn.Linear(EVENT_DIM, d_model)
        self.action_proj = nn.Linear(ACTION_FEATURE_DIM, d_model)

        # Learned per-slot positional embeddings distinguish the otherwise-identical action
        # tokens from each other (move slot 0 vs move slot 1 vs switch slot 3, ...) and give
        # the model a stable "address" for each output head. History tokens get relative
        # position embeddings too so ordering (not just content) is visible.
        self.action_pos = nn.Parameter(torch.randn(N_ACTIONS, d_model) * 0.02)
        self.history_pos = nn.Parameter(torch.randn(N_HISTORY, d_model) * 0.02)
        self.board_pos = nn.Parameter(torch.randn(1, d_model) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", norm_first=True, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.final_norm = nn.LayerNorm(d_model)

        self.policy_head = nn.Linear(d_model, 1)          # applied per action token
        self.value_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(),
                                         nn.Linear(d_model // 2, 1))
        self.opp_action_head = nn.Linear(d_model, N_ACTIONS)      # from board token only

        # Phase-3 aux heads, over the opponent's (up to 6) bench slots. Since this phase
        # doesn't yet have per-opponent-mon tokens, they read off the board token; a real
        # per-mon prediction (matching Jaxcalibur's "maintains predictions across turns")
        # wants per-mon tokens, which is the natural Phase-2/3 follow-up to this file.
        self.item_head = nn.Linear(d_model, N_OPP_BENCH * N_ITEM_CLASSES)
        self.ability_head = nn.Linear(d_model, N_OPP_BENCH * N_ABILITY_CLASSES)
        self.tera_head = nn.Linear(d_model, N_OPP_BENCH * N_TERA_CLASSES)

    def forward(self, board, history, action_feats, action_mask):
        """
        board:         float[B, N_VALUE_FEATURES]
        history:       float[B, N_HISTORY, EVENT_DIM]
        action_feats:  float[B, N_ACTIONS, ACTION_FEATURE_DIM]
        action_mask:   bool [B, N_ACTIONS]   (True = legal this turn)

        -> dict(policy_logits[B,N_ACTIONS] (illegal slots = -inf), value[B],
                opp_action_logits[B,N_ACTIONS], item_logits[B,6,N_ITEM_CLASSES],
                ability_logits[B,6,N_ABILITY_CLASSES], tera_logits[B,6,N_TERA_CLASSES])
        """
        b = self.board_proj(board).unsqueeze(1) + self.board_pos          # [B,1,D]
        h = self.history_proj(history) + self.history_pos.unsqueeze(0)    # [B,16,D]
        a = self.action_proj(action_feats) + self.action_pos.unsqueeze(0)  # [B,16,D]

        tokens = torch.cat([b, h, a], dim=1)                              # [B,1+16+16,D]
        # Padding-mask real history: an all-zero history row is a genuine "no event here"
        # padding slot (see history.history_window), not a legal token to attend from.
        hist_pad = (history.abs().sum(-1) == 0)                           # [B,16]
        key_padding_mask = torch.cat(
            [torch.zeros(board.shape[0], 1, dtype=torch.bool, device=board.device),
             hist_pad,
             torch.zeros(board.shape[0], a.shape[1], dtype=torch.bool, device=board.device)],
            dim=1,
        )
        enc = self.final_norm(self.encoder(tokens, src_key_padding_mask=key_padding_mask))

        board_tok = enc[:, 0]
        action_toks = enc[:, 1 + N_HISTORY:]

        policy_logits = self.policy_head(action_toks).squeeze(-1)          # [B,16]
        # A finite large-magnitude sentinel rather than -inf: softmax/log_softmax still send
        # these slots' probability to (exactly, in float32) 0, but -inf here combined with a
        # downstream p*log(p)-style computation (see train_policy.py's entropy term) produces
        # 0 * -inf = NaN gradients even when the offending element is masked out of the loss
        # afterward. -1e9 avoids the whole class of bug at essentially no cost in accuracy.
        policy_logits = policy_logits.masked_fill(~action_mask, -1e9)

        value = torch.tanh(self.value_head(board_tok).squeeze(-1))         # [-1,1]
        opp_action_logits = self.opp_action_head(board_tok)                # [B,16], unmasked:
        # the opponent's own legal-action mask isn't visible to us in general, so this head
        # is trained/evaluated against the opponent's REVEALED choice, not renormalized over
        # a mask we don't have.

        item_logits = self.item_head(board_tok).view(-1, N_OPP_BENCH, N_ITEM_CLASSES)
        ability_logits = self.ability_head(board_tok).view(-1, N_OPP_BENCH, N_ABILITY_CLASSES)
        tera_logits = self.tera_head(board_tok).view(-1, N_OPP_BENCH, N_TERA_CLASSES)

        return {
            "policy_logits": policy_logits,
            "value": value,
            "opp_action_logits": opp_action_logits,
            "item_logits": item_logits,
            "ability_logits": ability_logits,
            "tera_logits": tera_logits,
        }

    @torch.no_grad()
    def policy_prior(self, board, history, action_feats, action_mask):
        """Inference convenience: masked softmax over policy_logits only. Single example (no
        batch dim) in, [N_ACTIONS] numpy-friendly tensor out."""
        out = self.forward(board.unsqueeze(0), history.unsqueeze(0),
                            action_feats.unsqueeze(0), action_mask.unsqueeze(0))
        return torch.softmax(out["policy_logits"][0], dim=-1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
