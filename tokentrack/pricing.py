"""
Price table: USD per 1 million tokens.

This is the one file you will edit most often. Providers change prices and
add models constantly. Keep it as plain data so it's easy to update.

Each entry: model_name -> (input_price, output_price, cached_input_price)
cached_input_price is what you pay for input tokens served from the
provider's prompt cache (both OpenAI and Anthropic discount these).

Prices below were accurate as of late 2026 — verify against the provider
pricing pages before trusting a dollar figure, and update as needed.
"""

PRICES = {
    # ---- OpenAI ----
    "gpt-5": (1.25, 10.00, 0.125),
    "gpt-5-mini": (0.25, 2.00, 0.025),
    "gpt-5-nano": (0.05, 0.40, 0.005),
    "gpt-4.1": (2.00, 8.00, 0.50),
    "gpt-4.1-mini": (0.40, 1.60, 0.10),
    "gpt-4.1-nano": (0.10, 0.40, 0.025),
    "gpt-4o": (2.50, 10.00, 1.25),
    "gpt-4o-mini": (0.15, 0.60, 0.075),
    # ---- Anthropic ----
    "claude-opus-4-1": (15.00, 75.00, 1.50),
    "claude-opus-4": (15.00, 75.00, 1.50),
    "claude-sonnet-4-5": (3.00, 15.00, 0.30),
    "claude-sonnet-4": (3.00, 15.00, 0.30),
    "claude-haiku-4-5": (1.00, 5.00, 0.10),
    "claude-3-5-haiku": (0.80, 4.00, 0.08),
    # ---- Groq (free tier shows $0 on your bill; these are their paid list prices
    #      so you can see what the same usage *would* cost) ----
    "openai/gpt-oss-20b": (0.075, 0.30, 0.075),
    "openai/gpt-oss-120b": (0.15, 0.60, 0.15),
    # ---- Google Gemini (via OpenAI-compatible endpoint) ----
    "gemini-2.5-flash": (0.30, 2.50, 0.03),
    "gemini-2.5-pro": (1.25, 10.00, 0.125),
}


def lookup(model: str):
    """
    Find the price row for a model name.

    Providers return names like "gpt-4o-mini-2024-07-18" or
    "claude-sonnet-4-5-20250929". We match the longest known prefix so
    dated variants still resolve. Returns None if unknown.
    """
    if not model:
        return None
    model = model.lower()
    best = None
    for name, row in PRICES.items():
        if model.startswith(name) and (best is None or len(name) > len(best[0])):
            best = (name, row)
    return best[1] if best else None


def cost_usd(model: str, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> float | None:
    """
    Compute cost in dollars. Returns None if the model isn't in the table,
    so unknown models show up as "unpriced" instead of silently costing $0.
    """
    row = lookup(model)
    if row is None:
        return None
    in_price, out_price, cached_price = row
    # cached tokens are a subset of input tokens, billed at the cheaper rate
    uncached_input = max(input_tokens - cached_tokens, 0)
    dollars = (
        uncached_input * in_price
        + cached_tokens * cached_price
        + output_tokens * out_price
    ) / 1_000_000
    return round(dollars, 8)
