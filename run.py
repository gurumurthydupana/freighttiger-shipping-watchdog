"""Pipeline: shipments -> weekly cost/tonne-km -> baselines -> flags -> note retrieval -> verdict -> reasons -> CSV."""
import argparse, json
from pathlib import Path

import pandas as pd

from core import add_baselines, load_shipments, weekly_table
from explain import Explainer, TokenLog
from notes import Retriever, load_notes
from verdict import decide, is_flagged

COLUMNS = ["route", "week_of", "cost_per_tonne_km", "vs_own_history", "vs_similar_routes",
           "flagged", "matched_note_id", "reason"]


def run(shipments, notes_path, out, use_llm=True, fresh=False, log_path="logs/token_log.json"):
    weekly = add_baselines(weekly_table(load_shipments(shipments)))
    retriever = Retriever(load_notes(notes_path))
    log = TokenLog()
    explainer = Explainer(use_llm, fresh, log)
    rows = []
    for _, r in weekly.iterrows():
        if not is_flagged(r["vs_own_pct"], r["vs_peer_pct"]):
            continue
        v = decide(r["route"], r["week_of"].date(), r["vs_own_pct"], r["vs_peer_pct"], retriever)
        rows.append({
            "route": r["route"], "week_of": str(r["week_of"].date()),
            "cost_per_tonne_km": f"{r['cost_per_tonne_km']:.2f}",
            "vs_own_history": f"{r['vs_own_pct']:+.1f}% vs this route's past average",
            "vs_similar_routes": f"{r['vs_peer_pct']:+.1f}% vs similar-length routes this week",
            "flagged": "No (justified)" if v.justified else "Yes",
            "matched_note_id": v.note.note_id if v.note else "",
            "reason": explainer.reason(r, v)})
    out_df = pd.DataFrame(rows, columns=COLUMNS)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    explainer.save_cache()
    mode = "gemini" if explainer.key else "offline-template (no GEMINI_API_KEY set)"
    rep = log.report(mode) | {"output_rows": len(out_df), "llm_output_rejected_by_validator": explainer.fallbacks}
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    Path(log_path).write_text(json.dumps(rep, indent=2))
    return out_df, rep


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--shipments", default="data/shipment_records.csv")
    p.add_argument("--notes", default="data/context_notes.csv")
    p.add_argument("--out", default="output/flags.csv")
    p.add_argument("--no-llm", action="store_true", help="template reasons only (no API calls)")
    p.add_argument("--fresh", action="store_true", help="ignore the response cache (use for the token-cost run)")
    a = p.parse_args()
    df, rep = run(a.shipments, a.notes, a.out, not a.no_llm, a.fresh)
    print(df[["route", "week_of", "flagged", "matched_note_id"]].to_string(index=False))
    print(json.dumps(rep, indent=2))
