"""
Example: the same multi-turn chat against Anthropic, logged by tokentrack.

Run:  export ANTHROPIC_API_KEY=sk-ant-...
      python examples/anthropic_chat.py
"""
from anthropic import Anthropic
import tokentrack

client = tokentrack.wrap(Anthropic(), tag="example-anthropic")
conv = tokentrack.new_conversation()

history = []
for question in [
    "In one sentence, what is a token in an LLM?",
    "Give me a two-line example.",
    "Now say it like a pirate.",
]:
    history.append({"role": "user", "content": question})
    resp = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=200,
        messages=history,
        conversation_id=conv,
    )
    answer = resp.content[0].text
    history.append({"role": "assistant", "content": answer})
    print(f"\nQ: {question}\nA: {answer}")

print("\nDone. Events written to tokentrack.db — run: python report.py")
