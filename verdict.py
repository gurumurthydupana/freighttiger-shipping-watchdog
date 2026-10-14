"""Deterministic flag + verdict rules. No LLM is involved in any decision here."""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from notes import Note, Retriever

# Flag: cost is rising vs own history AND (a big jump, or a moderate jump that is also far above peers)
BIG_RISE_PCT = 10.0
MODERATE_RISE_PCT = 4.0
PEER_GAP_PCT = 12.0
MAG_TOLERANCE_PTS = 3.0  # observed rise may exceed a note's stated max by this much and still count as explained


def is_flagged(vs_own: float, vs_peer: float) -> bool:
    if vs_own != vs_own or vs_peer != vs_peer:  # NaN: no baseline to compare with
        return False
    return vs_own > 0 and (vs_own >= BIG_RISE_PCT or (vs_own >= MODERATE_RISE_PCT and vs_peer >= PEER_GAP_PCT))


@dataclass
class Verdict:
    justified: bool
    note: Optional[Note] = None
    rejected: List[Tuple[Note, str]] = field(default_factory=list)  # (note, reason code)


def decide(route, week_start, vs_own, vs_peer, retriever: Retriever) -> Verdict:
    inside, nearby = retriever.retrieve(route, week_start)
    v = Verdict(justified=False)
    for n in inside:
        if not n.supports_rise:
            v.rejected.append((n, "no_cost_effect"))
        elif n.scope == "All Routes" and vs_peer >= PEER_GAP_PCT:
            v.rejected.append((n, "peer_gap"))      # hit every route, so it cannot explain being above peers
        elif n.mag_hi is not None and vs_own > n.mag_hi + MAG_TOLERANCE_PTS:
            v.rejected.append((n, "too_large"))     # note explains a smaller rise than observed
        elif v.note is None:
            v.note, v.justified = n, True
        # extra matching notes are ignored: one exact citation is enough
    # notes just outside the week: say what they are about first (a "no effect" note never explains a rise)
    v.rejected += [(n, "no_cost_effect" if not n.supports_rise else "outside_window") for n in nearby]
    return v
