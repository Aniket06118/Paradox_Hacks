# main.py

from data_loader import load_data
from data_loader import validate_market_features
from segmentation import segment_by_column
from metric import compute_metrics
import json


def run_analysis(data_path):

    df = load_data(data_path)
    df = validate_market_features(df)

    overall = compute_metrics(df)

    segmentation = {
        "trend": segment_by_column(df, "trend"),
        "volatility": segment_by_column(df, "volatility"),
        "direction": segment_by_column(df, "direction"),
        "time_of_day": segment_by_column(df, "time_of_day_bucket"),
        "day_of_week": segment_by_column(df, "day_of_week"),
    }

    behavior = {
        "avg_holding_time": df["holding_time"].mean(),
        "avg_win_hold_time": df[df["pnl"] > 0]["holding_time"].mean(),
        "avg_loss_hold_time": df[df["pnl"] <= 0]["holding_time"].mean(),
    }

    final_report = {
        "overall": overall,
        "segmentation": segmentation,
        "behavior": behavior,
    }

    return final_report


if __name__ == "__main__":
    from llm_report import generate_report

    report = run_analysis(r"C:\Users\Aniket\Documents\AI-ML\Machine Learning\Hackathons\Paradox_Hacks\dummy_trades.csv")

    print(json.dumps(report, indent=4))

    # Generate LLM interpretation report
    print("\n" + "=" * 80)
    print("📋 LLM PERFORMANCE REPORT")
    print("=" * 80 + "\n")

    llm_output = generate_report(report)
    print(llm_output)