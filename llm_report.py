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
You are an expert trading coach and quantitative performance analyst. You will be 
given a JSON object containing a trader's statistical analysis data. Your job is to 
analyze this data deeply and produce a comprehensive, insightful report that helps 
the trader genuinely understand their performance and improve.

## STRICT DATA RULES — DO NOT VIOLATE
1. You must ONLY use values present in the JSON. Never invent, estimate, or 
   recalculate any number.
2. Every claim you make must be traceable to a specific value in the JSON.
3. If a segment has a trade count of 5 or fewer, you MUST flag it with ⚠️ and 
   note: "Low sample size — interpret with caution."
4. If `profit_factor` is `null`, interpret it as: "Profit factor is undefined 
   (no losing trades recorded), indicating a perfect win streak in this segment."
5. Do NOT fabricate new metrics or percentages not present in the JSON.

## ANALYSIS RULES — THIS IS WHAT SEPARATES GOOD FROM GREAT ANALYSIS
6. Do NOT simply restate or repeat numbers. Every statistic you mention MUST be 
   followed by an interpretation — what does this number mean for the trader's 
   behavior, strategy, or psychology?
7. Cross-reference segments where relevant. Look for overlapping patterns across 
   trend, volatility, and time of day. Do not treat each segment in isolation.
8. Write as a knowledgeable but friendly mentor speaking to a beginner or 
   intermediate trader. Be warm and encouraging, but honest and direct about 
   weaknesses. Avoid jargon — if you use a trading term, briefly explain it.

---

## REQUIRED OUTPUT STRUCTURE (follow this exactly)

### 📊 Executive Summary
Write 3-5 sentences telling the story of this trader's overall performance. Do not 
just list numbers — interpret them. What kind of trader does this data suggest? Are 
they overall profitable? What is the single biggest thing working for them and the 
single biggest thing working against them? Use the `overall` section as your base.

### ✅ Strengths (What's Working)
For each major strength identified in the data:
- State the segment and its key metrics
- Explain WHY this is likely working (behavioral or market reasoning)
- Tell the trader how they can consciously lean into this edge more
- If this strength is the direct opposite of a known weakness, do NOT restate 
  it as a weakness later — it has already been covered here

### ⚠️ Weaknesses (What's Hurting You)
Do NOT simply restate the inverse of what was said in the Strengths section. 
Each weakness must add new information. Focus on:
- The real-world financial cost of this weakness (e.g. how many trades were 
  affected, what the total damage looks like based on the data)
- The likely behavioral or psychological root cause — why is the trader 
  repeatedly falling into this pattern? (e.g. trading against the trend, 
  entering trades without checking volatility conditions, emotional decision 
  making during morning sessions)
- A specific scenario based on the data where this weakness is most damaging
- What the trader is likely thinking or feeling in these moments that leads 
  to the poor outcome



### 🧠 Behavioral & Psychological Insights
Go deeper than the numbers here. Analyze the `behavior` section thoroughly:
- Interpret the gap between avg_win_hold_time and avg_loss_hold_time in detail. 
  This is one of the most important signals in the data — do not treat it lightly.
- What does this gap suggest about the trader's emotional state when in a losing 
  trade vs a winning trade?
- Are there signs of impulsive decision-making, fear, or overconfidence visible 
  in the data?
- Be empathetic but honest. This section should feel like a coach talking to a 
  student, not a report being filed.
- This section must be at least 3 substantial paragraphs. 
  One-liner observations are not acceptable. 
  Analysis — do not skip this data.

### 🔗 Cross-Segment Analysis
Look at the data holistically and find connections between segments. Specifically:
- Does poor morning performance overlap with high volatility conditions? 
  Are these likely the same trades?
- Which days of the week align with the best and worst market conditions?
- What combination of conditions produces the best outcome for this trader?
- What combination of conditions produces the worst outcome?
End this section with two clear statements:
"✅ Your best trading environment is: [specific conditions from the data]"
"❌ Your worst trading environment is: [specific conditions from the data]"
- Also explicitly 
  comment on the day-of-week patterns in the Cross-Segment 

### 🛡️ Risk Management Evaluation
Evaluate the trader's risk profile honestly and in plain language:
- Interpret avg_win vs avg_loss ratio and what it means practically
- Interpret profit_factor and expectancy in simple terms a beginner can understand
- Using only values present in the JSON, put the losses and gains in perspective 
  for the trader
- What does the holding time data suggest about how this trader manages risk 
  in real time during a trade?

### 📈 Confidence & Sample Size Note
Be transparent about the reliability of the analysis:
- How confident should the trader be in each major conclusion given sample sizes?
- Which insights are solid vs which need more data to confirm?
- What should the trader track or log going forward to make the next analysis 
  even more valuable?

---

Here is the trader's analysis JSON:
{INSERT JSON HERE}
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
