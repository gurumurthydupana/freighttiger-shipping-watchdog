"""Plain-English reasons. Verdicts are fixed upstream; the LLM only phrases them, and its output is validated."""
import hashlib, json, os, re, time, urllib.error, urllib.request
from pathlib import Path

from notes import Note
from verdict import Verdict

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")  # check the current free-tier model name in AI Studio
URL = "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent"
PRICE_IN_PER_M, PRICE_OUT_PER_M = 0.30, 2.50  # USD per 1M tokens, paid-tier list price; verify. Free tier = $0.
CACHE_PATH = Path("cache/llm_cache.json")

SYSTEM = ("You write one or two plain-English sentences for a freight cost monitor. "
          "Use ONLY the facts in FACTS. Never add causes, dates, numbers or note ids that are not in FACTS. "
          "Do not change the verdict. If a note is listed as matched, name its id and date and restate what it says. "
          "If no note matched, say no supporting note was found, and if rejected notes are listed, name them and "
          "say briefly why they do not explain the rise.")

REJECT_TEXT = {
    "no_cost_effect": "it says costs were not affected / no disruption",
    "outside_window": "its window does not cover this week",
    "peer_gap": "it hit all routes, so it cannot explain being above similar routes",
    "too_large": "it explains a smaller rise than the one observed",
}


class TokenLog:
    def __init__(self):
        self.calls = self.tok_in = self.tok_out = self.cache_hits = 0
        self.est_in = self.est_out = 0  # character-based estimate, used when running without an API

    def report(self, mode):
        cost = self.tok_in / 1e6 * PRICE_IN_PER_M + self.tok_out / 1e6 * PRICE_OUT_PER_M
        return {"mode": mode, "model": MODEL, "llm_calls": self.calls, "cache_hits": self.cache_hits,
                "input_tokens": self.tok_in, "output_tokens": self.tok_out,
                "estimated_input_tokens_if_run": self.est_in, "estimated_output_tokens_if_run": self.est_out,
                "cost_usd_at_list_price": round(cost, 6), "cost_usd_on_free_tier": 0.0}


def facts_for(row, v: Verdict) -> dict:
    f = {"route": row["route"], "week_starting": str(row["week_of"].date()),
         "cost_per_tonne_km": round(row["cost_per_tonne_km"], 2),
         "rise_vs_own_history_pct": round(row["vs_own_pct"], 1), "rise_vs_similar_routes_pct": round(row["vs_peer_pct"], 1),
         "history_weeks_used": int(row["own_history_weeks_used"]),
         "verdict": "justified" if v.justified else "unexplained"}
    if v.note:
        f["matched_note"] = {"id": v.note.note_id, "date": str(v.note.date), "text": v.note.text}
    f["rejected_notes"] = [{"id": n.note_id, "date": str(n.date), "text": n.text, "why_rejected": REJECT_TEXT[c],
                            "note_window": f"{n.win_start} to {n.win_end}"} for n, c in v.rejected[:3]]
    return f


def template_reason(f: dict) -> str:
    if f["verdict"] == "justified":
        n = f["matched_note"]
        s = f"Matches note {n['id']} dated {n['date']}: {n['text']} The cost rise has a clear explanation."
    else:
        s = "No matching note found for this route or date range. Cost rise looks unexplained and worth a human review."
        if f["rejected_notes"]:
            parts = [f"{r['id']} ({r['date']}): {r['why_rejected']}" + (f" [{r['note_window']}]" if "window" in r["why_rejected"] else "")
                     for r in f["rejected_notes"][:2]]
            s = ("No note justifies this rise. Closest notes considered: " + "; ".join(parts) +
                 ". No genuine justification found; flagged for review.")
    if f["history_weeks_used"] < 8:
        s += f" (Own-history baseline uses only {f['history_weeks_used']} prior week(s).)"
    return s


def valid(text: str, f: dict) -> bool:
    """Reject LLM output that cites anything not in FACTS, or that omits the matched note id."""
    allowed = {f["matched_note"]["id"]} if "matched_note" in f else set()
    allowed |= {r["id"] for r in f["rejected_notes"]}
    cited = set(re.findall(r"\bN\d{3}\b", text))
    if not cited <= allowed:
        return False
    if f["verdict"] == "justified" and f["matched_note"]["id"] not in cited:
        return False
    if f["verdict"] == "unexplained" and re.search(r"clear explanation|justified by|explained by", text, re.I):
        return False
    return 20 < len(text) < 700


class Explainer:
    def __init__(self, use_llm: bool, fresh: bool, log: TokenLog):
        self.key = os.environ.get("GEMINI_API_KEY") if use_llm else None
        self.use_llm, self.fresh, self.log = use_llm, fresh, log
        self.cache = {} if fresh or not CACHE_PATH.exists() else json.loads(CACHE_PATH.read_text())
        self.fallbacks = 0

    def _call(self, prompt: str):
        body = {"systemInstruction": {"parts": [{"text": SYSTEM}]},
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0, "seed": 7, "maxOutputTokens": 800}}
        req = urllib.request.Request(URL.format(MODEL), json.dumps(body).encode(),
                                     {"Content-Type": "application/json", "x-goog-api-key": self.key})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = json.load(r)
                u = data.get("usageMetadata", {})
                self.log.calls += 1
                self.log.tok_in += u.get("promptTokenCount", 0)
                self.log.tok_out += u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()
            except urllib.error.HTTPError as e:
                print("Gemini error", e.code, e.read().decode()[:400])
                if e.code in (429, 500, 503):
                    time.sleep(2 ** attempt * 4)  # free tier rate limits
                    continue
                return None
        return None

    def reason(self, row, v: Verdict) -> str:
        f = facts_for(row, v)
        fallback = template_reason(f)
        prompt = "FACTS:\n" + json.dumps(f, indent=1)
        self.log.est_in += (len(SYSTEM) + len(prompt)) // 4
        self.log.est_out += 60
        if not self.key:
            return fallback
        h = hashlib.sha256((MODEL + SYSTEM + prompt).encode()).hexdigest()
        text = self.cache.get(h)
        if text:
            self.log.cache_hits += 1
        else:
            text = self._call(prompt)
            if text:
                self.cache[h] = text
        if not text or not valid(text, f):
            self.fallbacks += 1
            return fallback
        if f["history_weeks_used"] < 8:
            text += f" (Own-history baseline uses only {f['history_weeks_used']} prior week(s).)"
        return text

    def save_cache(self):
        if self.key:
            CACHE_PATH.parent.mkdir(exist_ok=True)
            CACHE_PATH.write_text(json.dumps(self.cache, indent=1, sort_keys=True))
