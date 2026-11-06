"""
Example: a tiny multi-turn chat with OpenAI, logged by tokentrack.

Run:  export OPENAI_API_KEY=sk-...   (Windows: set OPENAI_API_KEY=sk-...)
      python examples/openai_chat.py
"""
from openai import OpenAI
import tokentrack
import os 

# ---- the only tokentrack-specific lines are these two ----
client = tokentrack.wrap(
    OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url="https://api.groq.com/openai/v1"),
    tag="example-groq",
)
conv = tokentrack.new_conversation()
# -----------------------------------------------------------

history = []
for question in [
    "In one sentence, what is a token in an LLM?",
    "Give me a two-line example.",
    "Now say it like a pirate.",
]:
    history.append({"role": "user", "content": question})
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=history,
        conversation_id=conv,      # groups the three turns together
    )
    answer = resp.choices[0].message.content
    history.append({"role": "assistant", "content": answer})
    print(f"\nQ: {question}\nA: {answer}")

print("\nDone. Events written to tokentrack.db — run: python report.py")
