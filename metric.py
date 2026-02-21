import numpy as np

def compute_metrics(df):
    if len(df) == 0:
        return None

    total_trades = len(df)

    win_rate = (df["pnl"] > 0).mean()

    avg_win = df[df["pnl"] > 0]["pnl"].mean()
    avg_loss = abs(df[df["pnl"] <= 0]["pnl"].mean())

    expectancy = (win_rate * avg_win) - (
        (1 - win_rate) * avg_loss
    )

    total_profit = df[df["pnl"] > 0]["pnl"].sum()
    total_loss = abs(df[df["pnl"] <= 0]["pnl"].sum())

    profit_factor = (
        total_profit / total_loss if total_loss != 0 else np.inf
    )

    return {
        "total_trades": total_trades,
        "win_rate": round(win_rate, 3),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "expectancy": round(expectancy, 2),
        "profit_factor": round(profit_factor, 2),
    }