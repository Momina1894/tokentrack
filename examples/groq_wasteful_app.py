"""
Deliberately wasteful usage on Groq's free tier, so coach.py has something
real to find. Takes about a minute. All calls are logged to tokentrack.db.

    export GROQ_API_KEY=gsk_...
    python examples/groq_wasteful_app.py
    python coach.py
"""
from openai import OpenAI
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import tokentrack

key = os.environ.get("GROQ_API_KEY")
if not key:
    sys.exit("GROQ_API_KEY is not set in this terminal.")

client = tokentrack.wrap(OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1"))

SMALL = "openai/gpt-oss-20b"
BIG = "openai/gpt-oss-120b"

# A "document" the user pasted once and then keeps chatting about.
DOC = ("Quarterly report summary. " * 60 +
       "Revenue grew 12% year over year. Churn fell to 3.1%. Headcount is 48. "
       "The main risk flagged is dependence on two enterprise customers. " * 10)


def pause():
    time.sleep(1.2)  # stay well under the free-tier rate limit


# ---- 1. Context bloat: a 10-turn chat that re-sends the document every turn ----
print("1/3  long chat about a pasted document...")
conv = tokentrack.new_conversation()
history = []
for q in [
    "Hi, I'm going to paste a report and ask about it. Ready?",
    f"Here is the document:\n\n{DOC}\n\nWhat was the revenue growth?",
    "What is the churn rate?",
    "How many employees?",
    "What's the main risk?",
    "Summarise it in one line.",
    "Now in three bullet points.",
    "Rewrite the summary for an investor.",
    "Rewrite it for a new employee.",
    "What questions should the board ask?",
    "Thanks, give me a one-word mood for this report.",
]:
    history.append({"role": "user", "content": q})
    r = client.chat.completions.create(model=SMALL, messages=history, conversation_id=conv, tag="doc-chat")
    history.append({"role": "assistant", "content": r.choices[0].message.content})
    print("   turn done:", r.usage.prompt_tokens, "input tokens")
    pause()

# ---- 2. Model mismatch: tiny classification jobs on the BIG model ----
print("2/3  short classification calls on the big model...")
for text in [
    "I love this product!", "Terrible service, never again.", "It's okay I guess.",
    "Shipping was fast.", "The app crashes constantly.", "Five stars.",
    "Refund please.", "Works as described.", "Meh.", "Best purchase this year.",
]:
    client.chat.completions.create(
        model=BIG, tag="sentiment",
        messages=[{"role": "user", "content": f"Classify as positive/negative/neutral. One word only.\n\n{text}"}],
    )
    pause()

# ---- 3. Verbose output: short questions with no length instruction ----
print("3/3  short questions that invite long answers...")
for q in ["Explain photosynthesis.", "What is inflation?", "How do vaccines work?",
          "Why is the sky blue?", "What is a neural network?", "Explain compound interest.",
          "What causes earthquakes?", "How does GPS work?", "What is DNA?", "Explain supply and demand."]:
    client.chat.completions.create(model=SMALL, tag="faq-bot", messages=[{"role": "user", "content": q}])
    pause()

print("\nDone. Now run:  python coach.py")