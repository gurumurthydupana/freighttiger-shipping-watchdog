"""Context notes: parsing into structured form, and retrieval (hard filters + TF-IDF ranking)."""
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

PERSIST_WEEKS = 8  # an "all routes, starting this week" price rise stays visible vs the 8-week baseline
NEARBY_DAYS = 14   # notes this close to a week (but outside it) are reported as "considered, rejected"
QUERY = "cost increase higher freight rates surcharge diesel toll flooding disruption detour"

# A note "supports a rise" only if it has a positive cue and no negating cue.
NEG_CUES = ["not significantly affected", "not part of this dataset", "no major disruptions",
            "no significant disruptions", "without a rate change", "remained stable",
            "remained normal", "returned to normal", "improved road conditions"]
POS_CUES = ["surcharge", "higher trip costs", "pushing up", "prices rose", "price rise"]

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_RANGE = re.compile(r"from (\w{3})\w* (\d{1,2}) to (\w{3})\w* (\d{1,2})", re.I)
_MAG = re.compile(r"(\d+)\s*-\s*(\d+)\s*%")


def monday(d: date) -> date:
    return d - timedelta(days=d.weekday())


@dataclass(frozen=True)
class Note:
    note_id: str
    date: date
    scope: str                  # "All Routes" or "Origin-Destination"
    text: str
    supports_rise: bool
    win_start: date
    win_end: date
    mag_lo: Optional[float] = None
    mag_hi: Optional[float] = None

    def applies_to_route(self, route: str) -> bool:
        return self.scope == "All Routes" or self.scope == route

    def overlaps(self, week_start: date) -> bool:
        return self.win_start <= week_start + timedelta(days=6) and self.win_end >= week_start

    def gap_days(self, week_start: date) -> int:
        """0 if the note window overlaps the week, else days between them."""
        if self.overlaps(week_start):
            return 0
        week_end = week_start + timedelta(days=6)
        return (self.win_start - week_end).days if self.win_start > week_end else (week_start - self.win_end).days


def parse_note(row) -> Note:
    d = pd.to_datetime(row["date"]).date()
    text = row["note"]
    low = text.lower()
    supports = (not any(c in low for c in NEG_CUES)) and any(c in low for c in POS_CUES)
    m = _RANGE.search(text)
    if m:  # explicit range, e.g. "from Feb 24 to Mar 8"
        s = date(d.year, _MONTHS[m.group(1).lower()], int(m.group(2)))
        e = date(d.year, _MONTHS[m.group(3).lower()], int(m.group(4)))
        if e < s:
            e = e.replace(year=e.year + 1)
        ws, we = s, e
    elif "starting this week" in low:  # open-ended price change
        ws = monday(d)
        we = ws + timedelta(weeks=PERSIST_WEEKS) - timedelta(days=1)
    else:  # single event: the week the note is dated
        ws = monday(d)
        we = ws + timedelta(days=6)
    mag = _MAG.search(text)
    return Note(row["note_id"], d, row["applies_to"].strip(), text, supports, ws, we,
                float(mag.group(1)) if mag else None, float(mag.group(2)) if mag else None)


def load_notes(path):
    df = pd.read_csv(path)
    return [parse_note(r) for _, r in df.iterrows()]


class Retriever:
    """Hard filters (route scope, time window) do the safety work; TF-IDF only orders what survives."""

    def __init__(self, notes):
        self.notes = notes
        self.vec = TfidfVectorizer(stop_words="english").fit([n.text for n in notes] + [QUERY])
        self.q = self.vec.transform([QUERY])
        self.sim = {n.note_id: float((self.vec.transform([n.text]) @ self.q.T).toarray()[0, 0]) for n in notes}

    def retrieve(self, route: str, week_start: date):
        """Returns (in_window, nearby): notes for this route/All Routes overlapping the week, and notes just outside it."""
        scoped = [n for n in self.notes if n.applies_to_route(route)]
        inside = [n for n in scoped if n.overlaps(week_start)]
        nearby = [n for n in scoped if not n.overlaps(week_start) and n.gap_days(week_start) <= NEARBY_DAYS]
        rank = lambda n: (-self.sim[n.note_id], n.note_id)  # deterministic tie-break
        return sorted(inside, key=rank), sorted(nearby, key=rank)
