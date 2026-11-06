"""
coach.py — the rule-based "waste finder". No AI involved; pure SQL + pandas.

    python coach.py           # analyse your real tokentrack.db
    python coach.py --demo    # build a fake month of usage (demo.db) and analyse that

Each rule is a function that takes the events DataFrame and returns a list of
Finding objects. Adding a rule = adding one function. The report at the end
is the thing a user would actually read.
"""
import argparse
import os
import random
import sqlite3
import sys
from dataclasses import dataclass, field

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tokentrack.pricing import PRICES, cost_usd  # noqa: E402

pd.set_option("display.width", 160)


# ----------------------------------------------------------------------------
# A finding = one thing we want to tell the user
# ----------------------------------------------------------------------------
@dataclass
class Finding:
    rule: str                 # short name, e.g. "context_bloat"
    severity: str             # "high" | "medium" | "low"
    title: str                # one line, human
    detail: str               # a paragraph, human
    est_savings_usd: float    # how much they'd have saved; 0 if not applicable
    evidence: pd.DataFrame = field(default_factory=pd.DataFrame)


# ----------------------------------------------------------------------------
# Rule 1: context bloat
# Long conversations re-send the whole history each turn, so input tokens
# climb steadily. Past a point, starting a fresh chat with a summary is cheaper.
# ----------------------------------------------------------------------------
def rule_context_bloat(df: pd.DataFrame, min_turns=6, growth_ratio=3.0) -> list[Finding]:
    convs = df[df.conversation_id.notna()].sort_values("ts").copy()
    if convs.empty:
        return []
    convs["turn"] = convs.groupby("conversation_id").cumcount() + 1

    g = convs.groupby("conversation_id")
    summary = pd.DataFrame({
        "turns": g.size(),
        "first_input": g.input_tokens.first(),
        "last_input": g.input_tokens.last(),
        "total_input": g.input_tokens.sum(),
        "total_cost": g.cost_usd.sum(),
        "model": g.model.first(),
    })
    summary["growth"] = summary.last_input / summary.first_input.clip(lower=1)
    bad = summary[(summary.turns >= min_turns) & (summary.growth >= growth_ratio)]
    if bad.empty:
        return []

    # Savings estimate: if the user had reset context every `min_turns` turns,
    # input per turn would have stayed near the early-turn average.
    early = convs[convs.turn <= 3].groupby("conversation_id").input_tokens.mean()
    bad = bad.assign(early_avg=early.reindex(bad.index))
    bad["wasted_input"] = (bad.total_input - bad.early_avg * bad.turns).clip(lower=0)
    bad["price_in"] = [(PRICES.get(_price_key(m)) or (0, 0, 0))[0] for m in bad.model]
    bad["saving"] = bad.wasted_input * bad.price_in / 1e6
    bad = bad.sort_values("saving", ascending=False)
    total_saving = bad.saving.sum()

    top = bad.head(5).reset_index()[["conversation_id", "model", "turns", "first_input", "last_input", "saving"]]
    top["saving"] = top.saving.round(2)
    return [Finding(
        rule="context_bloat",
        severity="high" if total_saving > 1 else "medium",
        title=f"{len(bad)} conversations where input grew ≥{growth_ratio:.0f}x because history is re-sent every turn",
        detail=(f"Across them, about {bad.wasted_input.sum():,.0f} input tokens were spent re-reading earlier turns. "
                f"Starting a fresh chat (or summarising the thread) every ~{min_turns} turns would have avoided most of it. "
                f"Worst offenders below."),
        est_savings_usd=total_saving,
        evidence=top,
    )]


# ----------------------------------------------------------------------------
# Rule 2: model mismatch
# Expensive model used for short, simple exchanges. We can't judge "simple"
# perfectly without reading the text, but tiny prompt + tiny answer is a strong
# signal. Savings = what the same tokens cost on the cheap sibling model.
# ----------------------------------------------------------------------------
CHEAPER_SIBLING = {
    "gpt-5": "gpt-5-mini", "gpt-4.1": "gpt-4.1-mini", "gpt-4o": "gpt-4o-mini",
    "claude-opus-4-1": "claude-sonnet-4-5", "claude-opus-4": "claude-sonnet-4-5",
    "claude-sonnet-4-5": "claude-haiku-4-5", "claude-sonnet-4": "claude-haiku-4-5",
    "openai/gpt-oss-120b": "openai/gpt-oss-20b", "gemini-2.5-pro": "gemini-2.5-flash",
}


def _price_key(model: str) -> str | None:
    model = (model or "").lower()
    keys = [k for k in PRICES if model.startswith(k)]
    return max(keys, key=len) if keys else None


def rule_model_mismatch(df: pd.DataFrame, max_input=400, max_output=150) -> list[Finding]:
    d = df.copy()
    d["key"] = d.model.map(_price_key)
    d["cheaper"] = d.key.map(CHEAPER_SIBLING)
    small = d[d.cheaper.notna() & (d.input_tokens <= max_input) & (d.output_tokens <= max_output)]
    if small.empty:
        return []

    small = small.assign(
        cheap_cost=[cost_usd(r.cheaper, r.input_tokens, r.output_tokens, r.cached_tokens) or 0 for r in small.itertuples()]
    )
    small["saving"] = small.cost_usd.fillna(0) - small.cheap_cost
    per_model = small.groupby(["key", "cheaper"]).agg(calls=("id", "count"), spent=("cost_usd", "sum"), saving=("saving", "sum")).reset_index()

    findings = []
    for r in per_model.sort_values("saving", ascending=False).itertuples():
        if r.calls < 5:
            continue
        findings.append(Finding(
            rule="model_mismatch",
            severity="high" if r.saving > 1 else "medium",
            title=f"{r.calls} short calls on {r.key} could run on {r.cheaper}",
            detail=(f"These calls had ≤{max_input} input and ≤{max_output} output tokens — classification, "
                    f"extraction, short Q&A territory. They cost ${r.spent:.2f}; the same tokens on "
                    f"{r.cheaper} would be ${r.spent - r.saving:.2f}. Try the cheaper model on a sample and compare quality."),
            est_savings_usd=r.saving,
        ))
    return findings


# ----------------------------------------------------------------------------
# Rule 3: verbose output
# Output tokens cost 4–8x input tokens. If answers are routinely long for
# short questions, adding "answer concisely" / max_tokens pays off immediately.
# ----------------------------------------------------------------------------
def rule_verbose_output(df: pd.DataFrame, ratio=3.0, min_calls=10) -> list[Finding]:
    d = df[(df.input_tokens > 0)].copy()
    d["ratio"] = d.output_tokens / d.input_tokens
    verbose = d[d.ratio >= ratio]
    if len(verbose) < min_calls:
        return []
    by_tag = verbose.groupby(verbose.tag.fillna("(untagged)")).agg(calls=("id", "count"), out=("output_tokens", "sum"), cost=("cost_usd", "sum"))
    by_tag = by_tag[by_tag.calls >= min_calls]
    findings = []
    for tag, r in by_tag.sort_values("cost", ascending=False).iterrows():
        # assume a concise instruction halves output length
        saving = r.cost * 0.5 * 0.8  # 80% of cost is output on these calls, half of that is saveable
        findings.append(Finding(
            rule="verbose_output",
            severity="medium",
            title=f"'{tag}': {int(r.calls)} calls where the answer was ≥{ratio:.0f}x longer than the prompt",
            detail=(f"{int(r.out):,} output tokens on these calls. Output is the expensive direction. "
                    f"Add 'Answer in N sentences' or set max_tokens; halving output would save about ${saving:.2f}."),
            est_savings_usd=saving,
        ))
    return findings


# ----------------------------------------------------------------------------
# Rule 4: repeated prefix / no caching
# Same long prompt start sent many times (a big system prompt, a pasted doc)
# with cached_tokens == 0 means the user is paying full price for the same
# text over and over. Both OpenAI and Anthropic cache automatically or with a
# one-line flag, at a 50–90% discount.
# ----------------------------------------------------------------------------
def rule_no_caching(df: pd.DataFrame, min_input=1500, min_calls=20) -> list[Finding]:
    big = df[(df.input_tokens >= min_input) & (df.cached_tokens == 0)]
    if len(big) < min_calls:
        return []
    by = big.groupby(["model", big.tag.fillna("(untagged)")]).agg(calls=("id", "count"), inp=("input_tokens", "sum"), cost=("cost_usd", "sum")).reset_index()
    by = by[by.calls >= min_calls]
    findings = []
    for r in by.sort_values("cost", ascending=False).itertuples():
        key = _price_key(r.model)
        if not key:
            continue
        in_price, _, cached_price = PRICES[key]
        # assume ~70% of each big prompt is a repeated prefix that could be cached
        saving = r.inp * 0.7 * (in_price - cached_price) / 1e6
        findings.append(Finding(
            rule="no_caching",
            severity="high" if saving > 1 else "medium",
            title=f"{r.calls} large prompts on {r.model} ('{r.tag}') with zero cached tokens",
            detail=(f"{r.inp:,} input tokens, none served from cache. If these share a long system prompt "
                    f"or document, prompt caching would bill the repeated part at ${cached_price}/M instead of "
                    f"${in_price}/M. Estimated saving if ~70% is cacheable: ${saving:.2f}."),
            est_savings_usd=saving,
        ))
    return findings


# ----------------------------------------------------------------------------
# Rule 5: unpriced models (housekeeping, not savings)
# ----------------------------------------------------------------------------
def rule_unpriced(df: pd.DataFrame) -> list[Finding]:
    un = df[df.cost_usd.isna()]
    if un.empty:
        return []
    models = un.model.value_counts()
    return [Finding(
        rule="unpriced_models",
        severity="low",
        title=f"{len(un)} calls have no price (models missing from pricing.py)",
        detail="Add these to tokentrack/pricing.py so cost totals are complete: " + ", ".join(f"{m} ({n})" for m, n in models.items()),
        est_savings_usd=0,
    )]


RULES = [rule_context_bloat, rule_model_mismatch, rule_verbose_output, rule_no_caching, rule_unpriced]


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------
def analyse(db_path: str) -> None:
    df = pd.read_sql("SELECT * FROM events ORDER BY ts", sqlite3.connect(db_path))
    if df.empty:
        print("No events yet.")
        return

    total = df.cost_usd.sum()
    days = max((pd.to_datetime(df.ts).max() - pd.to_datetime(df.ts).min()).days, 1)
    print("=" * 72)
    print(f"  tokentrack coach — {len(df):,} calls over {days} day(s), ${total:.2f} spent")
    print("=" * 72)

    findings: list[Finding] = []
    for rule in RULES:
        findings.extend(rule(df))

    if not findings:
        print("\nNothing to flag. Either you're efficient or there isn't enough data yet.\n")
        return

    order = {"high": 0, "medium": 1, "low": 2}
    findings.sort(key=lambda f: (order[f.severity], -f.est_savings_usd))
    saveable = sum(f.est_savings_usd for f in findings)

    print(f"\n  Estimated avoidable spend: ${saveable:.2f} of ${total:.2f} ({saveable / total:.0%})\n")
    for i, f in enumerate(findings, 1):
        tag = {"high": "!!", "medium": " !", "low": "  "}[f.severity]
        money = f"  (~${f.est_savings_usd:.2f})" if f.est_savings_usd else ""
        print(f"{tag} {i}. [{f.rule}] {f.title}{money}")
        print("      " + f.detail.replace("\n", "\n      "))
        if not f.evidence.empty and len(f.evidence) <= 12:
            print(f.evidence.to_string(index=False).replace("\n", "\n      ").rjust(0))
        print()


# ----------------------------------------------------------------------------
# Demo data: a plausible month for a small dev team, with waste baked in
# ----------------------------------------------------------------------------
def build_demo(path: str = "demo.db") -> str:
    if os.path.exists(path):
        os.remove(path)
    os.environ["TOKENTRACK_DB"] = path
    from tokentrack.storage import SQLiteStore
    store = SQLiteStore(path)
    rng = random.Random(7)
    base = pd.Timestamp("2026-10-01", tz="UTC")

    def ev(model, inp, out, cached=0, conv=None, tag=None, day=0, provider=None):
        store.write_event({
            "ts": (base + pd.Timedelta(days=day, minutes=rng.randint(0, 1439))).isoformat(),
            "provider": provider or ("anthropic" if "claude" in model else "openai"),
            "model": model, "input_tokens": inp, "output_tokens": out, "cached_tokens": cached,
            "cost_usd": cost_usd(model, inp, out, cached), "latency_ms": rng.randint(300, 3000),
            "conversation_id": conv, "tag": tag,
        })

    # (a) a coding assistant: long chats whose context balloons  -> context_bloat
    for c in range(40):
        turns = rng.randint(8, 25)
        ctx = rng.randint(800, 2000)
        conv = f"code{c:02d}"
        day = rng.randint(0, 29)
        for t in range(turns):
            out = rng.randint(150, 600)
            # turns must be in time order; write them minutes apart on one day
            store.write_event({
                "ts": (base + pd.Timedelta(days=day, hours=9, minutes=3 * t)).isoformat(),
                "provider": "anthropic", "model": "claude-sonnet-4-5",
                "input_tokens": ctx, "output_tokens": out, "cached_tokens": 0,
                "cost_usd": cost_usd("claude-sonnet-4-5", ctx, out), "latency_ms": rng.randint(800, 4000),
                "conversation_id": conv, "tag": "code-assistant",
            })
            ctx += out + rng.randint(100, 400)

    # (b) a classifier on GPT-4o that should be on mini  -> model_mismatch
    for _ in range(3000):
        ev("gpt-4o", rng.randint(80, 250), rng.randint(3, 15), tag="ticket-classifier", day=rng.randint(0, 29))

    # (c) a summariser with a 3k system prompt and no caching  -> no_caching
    for _ in range(600):
        ev("gpt-4.1", rng.randint(3000, 4500), rng.randint(200, 400), tag="doc-summariser", day=rng.randint(0, 29))

    # (d) a chatbot that rambles  -> verbose_output
    for _ in range(1500):
        ev("gpt-4o-mini", rng.randint(40, 120), rng.randint(400, 900), tag="faq-bot", day=rng.randint(0, 29))

    # (e) some well-behaved usage, and one unknown model
    for _ in range(200):
        ev("claude-haiku-4-5", rng.randint(200, 800), rng.randint(50, 300), tag="extraction", day=rng.randint(0, 29))
    for _ in range(8):
        ev("mystery-model-9000", 500, 200, tag="experiment", day=rng.randint(0, 29))
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="generate demo.db with a fake month of usage and analyse it")
    ap.add_argument("--db", default=os.environ.get("TOKENTRACK_DB", "tokentrack.db"))
    args = ap.parse_args()
    db = build_demo() if args.demo else args.db
    if not os.path.exists(db):
        sys.exit(f"No {db} found. Run an example first, or use --demo.")
    analyse(db)