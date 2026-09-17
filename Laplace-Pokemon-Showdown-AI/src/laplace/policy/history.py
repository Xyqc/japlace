r"""Compact history of the last N_HISTORY meaningful protocol events (Phase 2).

Jaxcalibur's write-up (https://jaxcalibur.github.io/, "Architecture") describes 16 tokens
"containing a history of the last 16 game events (moves/switch-ins)" alongside the static
board tokens, feeding the same transformer. Jaxcalibur's own event schema isn't published
beyond that one sentence, so this is our own compact encoding of the same idea, built from
what's already on hand: poke-env's `AbstractBattle._replay_data`, a list of every parsed
protocol line (`split_message`) the battle has seen -- "recent protocol/game events" per the
spec, and it needs no new wire parsing.

This does NOT replace `_opp_tracker` / `_opp_speed` / `_opp_item` / `_pending` /
`_futility_hist` -- those stay authoritative for the deterministic guards, which need exact
answers ("is this Pokemon Choice-locked: yes/no"), not a learned approximation. This is an
additional, unified signal for the neural model to find patterns like switch->move->pivot or
repeated setup that the handcrafted trackers don't summarize on their own.

Each event is encoded as a fixed-width float vector (EVENT_DIM) rather than a learned token
embedding over a message vocabulary, so a history slot can be fed into `net.PolicyNet` the
same way any other input token is (a linear projection). Swapping this for a learned
embedding table over (kind, move-id, species-id) is a real Phase-2 follow-up once there's a
large enough self-play dataset to justify the extra parameters; it is not done here.
"""

N_HISTORY = 16

# side one-hot(2: us/them) + kind one-hot(6) + tera flag + recency
_KINDS = ("move", "switch", "status", "boost", "faint", "other")
EVENT_DIM = 2 + len(_KINDS) + 1 + 1


def _side_of(split_message, player_role):
    """'p1a: ...' / 'p2a: ...' prefix on the relevant token -> 0 (us) / 1 (them) / None."""
    for tok in split_message:
        if isinstance(tok, str) and len(tok) >= 2 and tok[:2] in ("p1", "p2"):
            return 0 if tok[:2] == player_role else 1
    return None


def classify_event(split_message):
    """One protocol line -> (kind, tera_flag), or None if it isn't a "meaningful" event."""
    if len(split_message) < 2:
        return None
    tag = split_message[1]
    if tag == "move":
        return "move", 0.0
    if tag in ("switch", "drag"):
        return "switch", 0.0
    if tag in ("-status", "-curestatus", "-start", "-end"):
        return "status", 0.0
    if tag in ("-boost", "-unboost", "-setboost", "-clearboost"):
        return "boost", 0.0
    if tag == "faint":
        return "faint", 0.0
    if tag == "-terastallize":
        return "other", 1.0
    return None


def push_event(history_list, split_message, player_role, turn_now):
    """Append a meaningful event to a battle's rolling history (mutates in place, keeps only
    the most recent N_HISTORY). Call once per parsed protocol line from
    `EnginePlayer._handle_battle_message`, alongside the existing tracker updates."""
    classified = classify_event(split_message)
    if classified is None:
        return
    side = _side_of(split_message, player_role)
    if side is None:
        return
    kind, tera = classified
    history_list.append({"turn": turn_now, "side": side, "kind": kind, "tera": tera})
    del history_list[:-N_HISTORY]


def history_window(history_list, turn_now):
    """-> float[N_HISTORY][EVENT_DIM], most-recent-last, zero-padded at the front when the
    battle is younger than N_HISTORY events."""
    window = history_list[-N_HISTORY:]
    pad = N_HISTORY - len(window)
    rows = [[0.0] * EVENT_DIM for _ in range(pad)]
    for ev in window:
        vec = [0.0] * EVENT_DIM
        vec[ev["side"]] = 1.0
        vec[2 + _KINDS.index(ev["kind"])] = 1.0
        vec[2 + len(_KINDS)] = ev["tera"]
        age = max(0, turn_now - ev["turn"])
        vec[2 + len(_KINDS) + 1] = 1.0 / (1.0 + age)      # 1.0 = this turn, decaying
        rows.append(vec)
    return rows
