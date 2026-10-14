"""Eval harness. Exit code 1 if any check fails. Run: python eval.py"""
import csv, sys
from collections import defaultdict
from datetime import date, datetime, timedelta

import pandas as pd

import run as pipeline
from notes import Retriever, load_notes, parse_note
from verdict import decide

SHIP, NOTES, OUT = "data/shipment_records.csv", "data/context_notes.csv", "output/flags.csv"
failures = []


def check(ok, msg):
    print(("PASS  " if ok else "FAIL  ") + msg)
    if not ok:
        failures.append(msg)


out, _ = pipeline.run(SHIP, NOTES, OUT, use_llm=False)
out["week_of"] = out["week_of"].astype(str)

# ---- 1. Format contract -----------------------------------------------------------------------------
sample_cols = next(csv.reader(open("data/sample_output_format_v2.csv")))
check(list(out.columns) == sample_cols, "column names/order match sample_output_format_v2.csv")

# ---- 2. Independent recomputation (plain csv/datetime, no pandas groupby) ----------------------------
def independent(route, week):
    rows = list(csv.DictReader(open(SHIP)))
    cost, tk = defaultdict(float), defaultdict(float)
    for r in rows:
        d = datetime.strptime(r["shipment_date"], "%Y-%m-%d").date()
        mon = d - timedelta(days=d.weekday())
        key = (r["origin"] + "-" + r["destination"], mon)
        cost[key] += float(r["freight_cost_inr"])
        tk[key] += float(r["quantity_tonnes"]) * float(r["distance_km"])
    cur = cost[(route, week)] / tk[(route, week)]
    prior = sorted(w for (rt, w) in cost if rt == route and w < week)[-8:]
    own = sum(cost[(route, w)] / tk[(route, w)] for w in prior) / len(prior)
    rtype = next(r["route_type"] for r in rows if r["origin"] + "-" + r["destination"] == route)
    peers = {rt for rt in {k[0] for k in cost}
             if rt != route and next(r["route_type"] for r in rows if r["origin"] + "-" + r["destination"] == rt) == rtype}
    peer_vals = [cost[(p, week)] / tk[(p, week)] for p in peers if (p, week) in cost]
    return cur, (cur / own - 1) * 100, (cur / (sum(peer_vals) / len(peer_vals)) - 1) * 100

for route, wk in [("Delhi-Jaipur", date(2024, 11, 11)), ("Chennai-Bangalore", date(2025, 3, 3)), ("Mumbai-Pune", date(2025, 12, 22))]:
    cur, o, p = independent(route, wk)
    row = out[(out.route == route) & (out.week_of == str(wk))].iloc[0]
    ok = (f"{cur:.2f}" == row.cost_per_tonne_km and row.vs_own_history.startswith(f"{o:+.1f}%")
          and row.vs_similar_routes.startswith(f"{p:+.1f}%"))
    check(ok, f"independent recomputation matches pipeline: {route} {wk} ({cur:.2f}, {o:+.1f}%, {p:+.1f}%)")

# ---- 3. Brief's three sample rows: numbers must match the brief exactly ------------------------------
brief = {("Delhi-Jaipur", "2024-11-11"): ("4.17", "+35.5%", "+21.0%"),
         ("Ahmedabad-Mumbai", "2025-01-20"): ("3.29", "+29.5%", "+22.5%"),
         ("Mumbai-Pune", "2025-09-15"): ("3.98", "+9.2%", "+23.6%")}
for (route, wk), (c, o, p) in brief.items():
    r = out[(out.route == route) & (out.week_of == wk)].iloc[0]
    check((r.cost_per_tonne_km, r.vs_own_history.split()[0], r.vs_similar_routes.split()[0]) == (c, o, p),
          f"brief sample numbers reproduced: {route} {wk}")

# ---- 4. Labeled set ---------------------------------------------------------------------------------
labels = pd.read_csv("eval_labels.csv", keep_default_na=False)
ok_flag = ok_note = 0
for _, l in labels.iterrows():
    m = out[(out.route == l.route) & (out.week_of == l.week_of)]
    got_flag = "NOT_FLAGGED" if m.empty else m.iloc[0].flagged
    got_note = "" if m.empty else (m.iloc[0].matched_note_id if pd.notna(m.iloc[0].matched_note_id) else "")
    f_ok, n_ok = got_flag == l.expected_flagged, got_note == l.expected_note_id
    ok_flag += f_ok; ok_note += n_ok
    if not (f_ok and n_ok):
        check(False, f"label {l.route} {l.week_of}: expected {l.expected_flagged}/{l.expected_note_id or '-'}, got {got_flag}/{got_note or '-'}")
check(ok_flag == len(labels), f"labeled set verdicts: {ok_flag}/{len(labels)}")
check(ok_note == len(labels), f"labeled set matched_note_id: {ok_note}/{len(labels)}")

# ---- 5. Hallucination audit: every justified row must sit inside a hand-written legitimate window -------
# Written from reading the notes by hand, NOT derived from the parser.
LEGIT = {"N001": ("Chennai-Bangalore", date(2025, 2, 24), date(2025, 3, 8)),
         "N002": ("Ahmedabad-Mumbai", date(2025, 1, 20), date(2025, 1, 26)),
         "N003": (None, date(2025, 5, 5), date(2025, 6, 29))}
just = out[out.flagged == "No (justified)"]
bad = 0
for _, r in just.iterrows():
    rt, s, e = LEGIT.get(r.matched_note_id, (None, None, None))
    wk = date.fromisoformat(r.week_of)
    inside = s is not None and (rt in (None, r.route)) and wk <= e and wk + timedelta(days=6) >= s
    cites = r.matched_note_id in r.reason
    bad += not (inside and cites)
check(bad == 0 and len(just) > 0, f"all {len(just)} justified rows cite a note that fits route+window, and the reason names it")
check(all((out.flagged == "No (justified)") == (out.matched_note_id.notna() & (out.matched_note_id != ""))),
      "matched_note_id is filled exactly when verdict is justified")

# ---- 6. Adversarial notes: none of these may ever justify a rise -------------------------------------
base = dict(note_id="X1", date="2025-01-20", applies_to="Ahmedabad-Mumbai",
            note="A festival week saw a temporary surcharge applied by transporters on this corridor.")
mk = lambda **kw: Retriever([parse_note({**base, **kw})])
wk = date(2025, 1, 20)
cases = [("correct route + window (positive control)", mk(), "Ahmedabad-Mumbai", wk, 29.5, 22.5, True),
         ("right text, wrong route", mk(applies_to="Delhi-Jaipur"), "Ahmedabad-Mumbai", wk, 29.5, 22.5, False),
         ("right route, window 3 weeks later", mk(date="2025-02-10"), "Ahmedabad-Mumbai", wk, 29.5, 22.5, False),
         ("right route+window, note says no cost effect",
          mk(note="Festival week; costs were not significantly affected on this corridor."), "Ahmedabad-Mumbai", wk, 29.5, 22.5, False),
         ("note says returned to normal", mk(note="Corridor returned to normal after the festival surcharge ended."), "Ahmedabad-Mumbai", wk, 29.5, 22.5, False),
         ("diesel 5-7% note vs observed 20% rise", mk(applies_to="All Routes", note="Diesel prices rose nationwide starting this week, pushing up costs by roughly 5-7%."), "Ahmedabad-Mumbai", wk, 20.0, 3.0, False),
         ("diesel note cannot explain a gap vs peers", mk(applies_to="All Routes", note="Diesel prices rose nationwide starting this week, pushing up costs by roughly 5-7%."), "Ahmedabad-Mumbai", wk, 6.0, 25.0, False),
         ("diesel note, in-range rise shared with peers (positive control)", mk(applies_to="All Routes", note="Diesel prices rose nationwide starting this week, pushing up costs by roughly 5-7%."), "Ahmedabad-Mumbai", wk, 6.0, 4.0, True)]
for name, rt, route, w, o, p, expect in cases:
    check(decide(route, w, o, p, rt).justified == expect, f"adversarial: {name} -> {'justified' if expect else 'unexplained'}")

# ---- 7. Threshold sensitivity (informational) --------------------------------------------------------
import verdict
w = pipeline.add_baselines(pipeline.weekly_table(pipeline.load_shipments(SHIP))).dropna(subset=["vs_own_pct", "vs_peer_pct"])
print("\nThreshold sensitivity (flagged rows):")
for big, mod, gap in [(10, 4, 12), (8, 4, 12), (12, 4, 12), (10, 3, 12), (10, 5, 12), (10, 4, 10), (10, 4, 15)]:
    verdict.BIG_RISE_PCT, verdict.MODERATE_RISE_PCT, verdict.PEER_GAP_PCT = big, mod, gap
    n = sum(verdict.is_flagged(a, b) for a, b in zip(w.vs_own_pct, w.vs_peer_pct))
    print(f"  big>={big} moderate>={mod} peer>={gap}: {n}")

print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} CHECK(S) FAILED"))
sys.exit(1 if failures else 0)
