# Shipping Cost Watchdog (FreightTiger case study)

Finds routes whose cost per tonne-km is rising in a way that looks out of the ordinary, then checks a small set
of context notes to see whether there is a real, documented reason. Anything it cannot support with a note is
reported as **unexplained** rather than guessed.

## Run it
```bash
pip install -r requirements.txt
export GEMINI_API_KEY=...            # free key from Google AI Studio; optional (see below)
python run.py --fresh                # writes output/flags.csv and logs/token_log.json
python eval.py                       # eval harness (exit code 1 on any failure)
python repro_check.py                # three runs, compares them
```
Without a key (or with `--no-llm`) the same pipeline runs with template reasons and makes zero API calls.
Verdicts and cited notes are identical either way.

## How it works
| Stage | File | Uses an LLM? |
|---|---|---|
| Weekly cost per tonne-km, baselines | `core.py` | No |
| Flag rule | `verdict.py` | No |
| Note parsing + retrieval | `notes.py` | No (TF-IDF ranking only) |
| Justified / unexplained verdict | `verdict.py` | **No** |
| Plain-English reason | `explain.py` | Yes (Gemini, validated) |

**Design principle:** the LLM never decides anything. Every number, flag, verdict and cited note comes from
deterministic code. Gemini only rephrases a verdict that is already fixed, from a small JSON of facts, and its
output is rejected (and replaced by a template) if it cites a note id that is not in those facts, omits the
matched note id, or calls an unexplained row "justified". That is what makes hallucinated reasons and run-to-run
drift structurally hard.

### Core logic (matches the brief's definitions exactly)
- Route = `origin-destination`; `route_type` is used as given.
- Weeks run Monday-Sunday; `week_of` is the Monday.
- Cost per tonne-km = sum(freight cost) / sum(quantity x distance) for the route-week (ratio of totals, not an average of per-shipment ratios).
- **vs own history:** mean of the route's previous up-to-8 weekly values, strictly before the current week.
  With fewer than 8 prior weeks, all available prior weeks are used (no padding, no extrapolation); the reason text says so.
  The first week of each route has no history, so it can never be flagged (7 rows).
- **vs similar routes:** mean cost per tonne-km that week of the *other* routes with the same `route_type`.
- All three sample rows in the brief are reproduced exactly (cost, vs-own, vs-peer).

### Flag rule
Flagged if cost is rising vs own history **and** either (a) rise >= 10%, or (b) rise >= 4% and >= 12% above similar routes.
Chosen from the data: normal week-to-week noise is about +/-4% (std 4.2%); the 99th percentile of vs-own is +11%.
`eval.py` prints how the flag count moves with each threshold (20-32 rows across reasonable settings).
Note: the Short group has only two routes, so each one's "peer average" is the other route; peer comparisons there are weak.

### Retrieval and verdict
1. Each note is parsed into scope (route or All Routes), a time window, whether it supports a cost rise, and any stated % range.
   Windows: explicit "from X to Y" ranges are used as written; "starting this week" notes run 8 weeks (the same length as the baseline window, after which the new level is part of the baseline); other notes cover the week they are dated.
2. For a flagged row, notes are retrieved with **hard filters first** (applies to this route or All Routes; window overlaps this week), then ordered by TF-IDF similarity.
3. A note justifies a row only if it is a real cost driver (no "not affected / returned to normal / stable / improved" language) and:
   - an All-Routes note cannot explain a gap vs peers (it hit them too), and
   - if the note states a magnitude (e.g. 5-7%), the observed rise may not exceed the stated max by more than 3 points.
4. Otherwise the row is unexplained and `matched_note_id` is blank. The reason names the closest notes that were considered and why they were rejected.

Judgment calls worth discussing: Mumbai-Delhi 2025-05-12 (+11.6% vs N003's 5-7%) and Mumbai-Pune in the diesel window (large gap vs peers) are marked unexplained. I chose to under-claim; the brief says a confident wrong reason scores worse than "couldn't find one".

## How I checked it (`eval.py`)
1. Column names/order match `sample_output_format_v2.csv`.
2. Three route-weeks are recomputed with a separate from-scratch implementation (plain `csv` + `datetime`, no pandas groupby) and must match the pipeline.
3. The brief's 3 sample rows are reproduced exactly (numbers).
4. 16 labeled rows (3 from the brief, 13 hand-labeled from reading the notes, including negatives and the "window just ended" and "back to normal" traps): 16/16.
   Caveat: I wrote the hand labels after exploring the data, so this is a regression and sanity check, not an unbiased accuracy estimate. The two judgment-call labels are marked in the file.
5. Hallucination audit: every justified row must fall inside a hand-written legitimate (note, route, dates) table, and its reason must name the note.
6. Eight adversarial cases with synthetic notes (wrong route, wrong window, "no effect" text, "returned to normal", magnitude mismatch, all-route note vs peer gap), each with expected outcome, including positive controls so the guard is not just rejecting everything.

## Reproducibility
`repro_check.py` runs the pipeline three times (`--fresh`, no cache) and compares all non-reason columns (hash) plus the reason strings.
Randomness: none in the pipeline. The Gemini call uses `temperature=0, seed=7`; verdicts do not depend on it.
Responses are cached by prompt hash in `cache/` so normal reruns are identical; use `--fresh` for the cost run and the repro check.
Result: 3/3 identical on flags, numbers, verdicts, note ids; 0 reason strings differed in wording only.

## Cost log (one full run)
See `logs/token_log.json` (real counts from Gemini's `usageMetadata`). One LLM call per output row (24 rows), no retries beyond rate limits.
Offline estimate: about 5.2k input and 1.4k output tokens in total. Free tier cost: $0. At list price this is well under one US cent.
Real run (`gemini-3.5-flash-lite`): llm_calls=24, input_tokens=6981, output_tokens=1672, cost_usd_at_list_price=0.006274. `cost_usd_at_list_price` is an estimate using placeholder rates; actual cost on the free tier is 0.

## Limitations
- Notes are interpreted with cue-word rules (transparent and deterministic, but built around this note style); a very different note vocabulary would need new cues. An LLM classifier could replace this at the cost of reproducibility.
- Retrieval over 10 notes does not need a vector database; hard filters + TF-IDF are enough. At thousands of notes I would add embeddings and keep the same hard filters.
- Not built: the natural-language Q&A stretch goal.
