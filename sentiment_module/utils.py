# utils.py

import pandas as pd
import matplotlib.pyplot as plt

def extract_sentiment_columns(df, col="sentiment_result"):
    df["sentiment_label"] = df[col].apply(lambda x: x["label"])
    df["sentiment_confidence"] = df[col].apply(lambda x: x["confidence"])
    df["score_negative"] = df[col].apply(lambda x: x["score_distribution"]["negative"])
    df["score_neutral"] = df[col].apply(lambda x: x["score_distribution"]["neutral"])
    df["score_positive"] = df[col].apply(lambda x: x["score_distribution"]["positive"])
    return df

def plot_sentiment_distribution(df):
    counts = df["sentiment_label"].value_counts()
    plt.figure(figsize=(6, 4))
    counts.plot(kind="bar", color=["#ef5350", "#ffee58", "#66bb6a"])
    plt.title("Sentiment Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

def generate_summary(df):
    summary = {
        "Total samples": len(df),
        "Positive %": round((df["sentiment_label"] == "positive").mean() * 100, 2),
        "Neutral %": round((df["sentiment_label"] == "neutral").mean() * 100, 2),
        "Negative %": round((df["sentiment_label"] == "negative").mean() * 100, 2),
        "High confidence max": df["sentiment_confidence"].max(),
        "Low confidence min": df["sentiment_confidence"].min(),
        "Low confidence count (< 0.6)": int((df["sentiment_confidence"] < 0.6).sum())
    }
    return pd.DataFrame([summary])
