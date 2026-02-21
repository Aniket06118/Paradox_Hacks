# llm_report.py — LLM Interpretation Layer for Trading Analytics
# Sends structured analytics JSON to GROQ API and returns a professional report.
# The LLM only INTERPRETS numbers — it never recalculates or fabricates data.

import os
import json

from dotenv import load_dotenv
from groq import Groq


# ---------------------------------------------------------------------------
# System prompt — defines the LLM's role, output structure, and constraints
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a professional quantitative trading performance analyst. Your role is to \
provide structured performance diagnostics based EXCLUSIVELY on the analytics JSON \
provided by the user.

## STRICT RULES — VIOLATION IS NOT ACCEPTABLE
1. You must ONLY use values present in the JSON. Never invent, estimate, or \
   recalculate any number.
2. Every claim you make must be traceable to a specific value in the JSON.
3. If a segment has a very low trade count (≤ 5), you MUST flag it with a \
   statistical caution note (e.g., "⚠️ Low sample size — interpret with caution").
4. If `profit_factor` is `null` (meaning zero losses), interpret it as: \
   "Profit factor is undefined (no losing trades recorded), indicating a perfect \
   win streak in this segment."
5. Do NOT give generic trading advice. Every recommendation must be grounded in \
   observed weaknesses from the data.
6. Do NOT fabricate new metrics or percentages not present in the JSON.

## REQUIRED OUTPUT STRUCTURE (follow this exactly)

### 📊 Executive Summary
Provide a concise 3–5 sentence overview of overall trading performance using the \
`overall` section metrics (total_trades, win_rate, avg_win, avg_loss, expectancy, \
profit_factor).

### ✅ Strengths
Bullet points highlighting the strongest performing segments. Reference specific \
segment names and their metrics (win_rate, expectancy, profit_factor).

### ⚠️ Weaknesses
Bullet points highlighting the weakest performing segments. Reference specific \
segment names and their metrics.

### 🧠 Behavioral Analysis
Interpret the `behavior` section:
- Average holding time overall
- Difference between winning and losing trade hold times
- What this implies about trading discipline and decision-making patterns

### 🛡️ Risk Management Evaluation
Analyze:
- avg_win vs avg_loss ratio (reward-to-risk)
- profit_factor interpretation
- expectancy interpretation (expected $ per trade)
- Overall risk profile assessment

### 🎯 Action Plan
Provide 3–5 numbered recommendations based STRICTLY on observed weaknesses. Each \
recommendation must reference specific data points from the JSON.

### 📈 Confidence Assessment
Comment on:
- Total number of trades in the dataset
- Whether the sample size is sufficient for statistically reliable conclusions
- Any segments with too-low sample sizes to draw firm conclusions
"""


def build_prompt(analytics_json: dict) -> str:
    """
    Serialize the analytics JSON into a structured user prompt.

    Args:
        analytics_json: The complete analytics output from the trading engine,
                        containing 'overall', 'segmentation', and 'behavior' keys.

    Returns:
        A string prompt ready to send as the user message to the LLM.
    """
    json_str = json.dumps(analytics_json, indent=2, default=str)

    user_prompt = (
        "Below is the complete analytics JSON produced by a deterministic trading "
        "analytics engine. Interpret these results and produce a professional "
        "trading performance report following the required structure.\n\n"
        "IMPORTANT REMINDERS:\n"
        "- Do NOT recalculate any metric. Use the values as-is.\n"
        "- Do NOT hallucinate or invent any data point.\n"
        "- Reference specific numbers when making claims.\n"
        "- Flag any segment with ≤ 5 trades as low sample size.\n\n"
        "```json\n"
        f"{json_str}\n"
        "```"
    )

    return user_prompt


def generate_report(
    analytics_json: dict,
    model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.3,
    max_tokens: int = 3000,
) -> str:
    """
    Generate a professional trading performance report from analytics JSON.

    This function:
      1. Loads the GROQ API key from .env
      2. Constructs a structured prompt using build_prompt()
      3. Sends the prompt to the GROQ API
      4. Returns the formatted text report

    Args:
        analytics_json: The complete analytics output dict.
        model:          GROQ model to use (default: llama-3.3-70b-versatile).
        temperature:    Sampling temperature (low = more deterministic).
        max_tokens:     Maximum response length.

    Returns:
        The LLM-generated report as a string.

    Raises:
        ValueError: If GROQ_API_KEY is not set.
        groq.APIError: On API communication failures.
    """
    # Load environment variables from .env file
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        raise ValueError(
            "GROQ_API_KEY is not set. "
            "Please add your key to the .env file in the project root."
        )

    client = Groq(api_key=api_key)

    user_prompt = build_prompt(analytics_json)

    chat_completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return chat_completion.choices[0].message.content
