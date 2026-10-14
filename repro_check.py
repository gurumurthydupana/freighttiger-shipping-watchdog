"""Three untouched runs on the same input. Numbers, flags, verdicts and cited notes must be identical;
reason wording may differ (only possible when an LLM is used). Run: python repro_check.py"""
import hashlib, sys
import pandas as pd
import run as pipeline

outs = []
for i in (1, 2, 3):
    df, rep = pipeline.run("data/shipment_records.csv", "data/context_notes.csv", f"output/repro_run{i}.csv",
                           use_llm=True, fresh=True, log_path=f"logs/repro_run{i}_tokens.json")
    outs.append(df.fillna(""))
fixed = [c for c in outs[0].columns if c != "reason"]
hashes = [hashlib.sha256(o[fixed].to_csv(index=False).encode()).hexdigest()[:16] for o in outs]
same_fixed = len(set(hashes)) == 1
reason_diffs = int(sum((outs[0].reason != outs[k].reason).sum() for k in (1, 2)))
print("hash of all non-reason columns per run:", hashes)
print("identical flags / numbers / verdicts / matched_note_id across 3 runs:", same_fixed)
print("reason strings that differ between run1 and runs 2/3:", reason_diffs, "(wording only)")
sys.exit(0 if same_fixed else 1)
