"""
Quick look at what's in tokentrack.db. This is the seed of the dashboard.

Run: python report.py
"""
import os
import sqlite3

import pandas as pd

db = os.environ.get("TOKENTRACK_DB", "tokentrack.db")
if not os.path.exists(db):
    raise SystemExit(f"No {db} yet — run one of the examples first.")

df = pd.read_sql("SELECT * FROM events ORDER BY ts", sqlite3.connect(db))
pd.set_option("display.width", 160)

print(f"\n{len(df)} calls logged, ${df.cost_usd.sum():.4f} total\n")

print("== By model ==")
print(
    df.groupby(["provider", "model"])
      .agg(calls=("id", "count"), input_tokens=("input_tokens", "sum"),
           output_tokens=("output_tokens", "sum"), cost_usd=("cost_usd", "sum"))
      .sort_values("cost_usd", ascending=False)
)

print("\n== Per-turn input tokens within each conversation (context bloat check) ==")
convs = df[df.conversation_id.notna()].copy()
convs["turn"] = convs.groupby("conversation_id").cumcount() + 1
print(convs[["conversation_id", "turn", "model", "input_tokens", "output_tokens", "cost_usd"]].to_string(index=False))

unpriced = df[df.cost_usd.isna()].model.unique()
if len(unpriced):
    print(f"\n(!) Models missing from pricing.py: {list(unpriced)}")
