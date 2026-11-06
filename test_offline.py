"""
Offline test: fakes the network so we can check the wrapper logs correctly
without spending any money. Run: python test_offline.py
"""
import os
import sqlite3
from types import SimpleNamespace

os.environ["TOKENTRACK_DB"] = "test_tokentrack.db"
if os.path.exists("test_tokentrack.db"):
    os.remove("test_tokentrack.db")

from openai import OpenAI
from anthropic import Anthropic
import tokentrack

# ---- fake OpenAI response shaped like the real SDK object ----
def fake_openai_create(**kwargs):
    return SimpleNamespace(
        model="gpt-4o-mini-2024-07-18",
        choices=[SimpleNamespace(message=SimpleNamespace(content="Hello from fake OpenAI"))],
        usage=SimpleNamespace(prompt_tokens=120, completion_tokens=30,
                              prompt_tokens_details=SimpleNamespace(cached_tokens=100)),
    )

# ---- fake Anthropic response ----
def fake_anthropic_create(**kwargs):
    return SimpleNamespace(
        model="claude-haiku-4-5-20251001",
        content=[SimpleNamespace(type="text", text="Hello from fake Anthropic")],
        usage=SimpleNamespace(input_tokens=200, output_tokens=50,
                              cache_read_input_tokens=0, cache_creation_input_tokens=0),
    )

oa = OpenAI(api_key="fake")
oa.chat.completions.create = fake_openai_create
oa = tokentrack.wrap(oa, tag="test", store_text=True)

an = Anthropic(api_key="fake")
an.messages.create = fake_anthropic_create
an = tokentrack.wrap(an, tag="test")

conv = tokentrack.new_conversation()
r1 = oa.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}], conversation_id=conv)
r2 = an.messages.create(model="claude-haiku-4-5", max_tokens=50, messages=[{"role": "user", "content": "hi"}], conversation_id=conv)

assert r1.choices[0].message.content == "Hello from fake OpenAI"   # response passed through untouched
assert r2.content[0].text == "Hello from fake Anthropic"

conn = sqlite3.connect("test_tokentrack.db")
rows = conn.execute(
    "select provider, model, input_tokens, output_tokens, cached_tokens, cost_usd, conversation_id, tag, prompt_text from events"
).fetchall()
conn.close()  # Windows won't delete a file that's still open
for r in rows:
    print(r)

assert len(rows) == 2
oa_row, an_row = rows
assert oa_row[2:5] == (120, 30, 100)
# 20 uncached @0.15 + 100 cached @0.075 + 30 out @0.60  per million
assert abs(oa_row[5] - (20*0.15 + 100*0.075 + 30*0.60) / 1e6) < 1e-12
assert an_row[2:5] == (200, 50, 0)
assert abs(an_row[5] - (200*1.00 + 50*5.00) / 1e6) < 1e-12
assert oa_row[6] == an_row[6] == conv
assert oa_row[8] is not None and an_row[8] is None  # store_text respected
print("\nAll checks passed.")
try:
    os.remove("test_tokentrack.db")
except PermissionError:
    pass  # leftover temp file; safe to delete by hand
