import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["entry_time", "exit_time"])

    # Ensure pnl exists
    df["pnl"] = (
        (df["exit_price"] - df["entry_price"]) * df["quantity"]
    )

    df.loc[df["direction"] == "SHORT", "pnl"] *= -1

    # Return percentage
    df["return_pct"] = df["pnl"] / (
        df["entry_price"] * df["quantity"]
    )

    # Outcome label
    df["outcome"] = df["pnl"].apply(
        lambda x: "WIN" if x > 0 else "LOSS"
    )

    return df

REQUIRED_MARKET_COLUMNS = [
    "trend",
    "volatility",
    "volume_level",
    "distance_from_ma",
    "rsi_value",
    "distance_from_recent_high",
    "distance_from_recent_low",
]


def validate_market_features(df):
    missing = [
        col for col in REQUIRED_MARKET_COLUMNS if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing market columns: {missing}")

    return df

# test_df=load_data(r"C:\Users\Aniket\Documents\AI-ML\Machine Learning\Hackathons\Paradox_Hacks\dummy_trades.csv")
# test_df=validate_market_features(test_df)
# pd.set_option('display.max_columns', None)
# print(test_df.head())