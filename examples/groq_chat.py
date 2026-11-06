"""
Example: a tiny multi-turn chat on Groq's FREE tier, logged by tokentrack.

Groq speaks the OpenAI API format, so we use the OpenAI client and just
point it at Groq's server.

1. Get a free key at https://console.groq.com  (no card needed)
2. In the terminal:
       export GROQ_API_KEY=gsk_...        (Mac/Linux)
       set GROQ_API_KEY=gsk_...           (Windows cmd)
       $env:GROQ_API_KEY="gsk_..."        (Windows PowerShell)
3. python examples/groq_chat.py
"""
import os
import sys

from openai import OpenAI
import tokentrack

key = os.environ.get("GROQ_API_KEY")
if not key:
    sys.exit("GROQ_API_KEY is not set in this terminal. See the top of this file.")

client = tokentrack.wrap(
    OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1"),
    tag="example-groq",
)
conv = tokentrack.new_conversation()

history = []
for question in [
    "In one sentence, what is a token in an LLM?",
    "Give me a two-line example.",
    "Now say it like a pirate.",
]:
    history.append({"role": "user", "content": question})
    resp = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=history,
        conversation_id=conv,
    )
    answer = resp.choices[0].message.content
    history.append({"role": "assistant", "content": answer})
    print(f"\nQ: {question}\nA: {answer}")

print("\nDone. Events written to tokentrack.db — run: python report.py")
