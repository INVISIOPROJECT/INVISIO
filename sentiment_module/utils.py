import matplotlib.pyplot as plt
import pandas as pd

def extract_sentiment_columns(df):
    # نبحث عن العمود اللي فيه dict مع label و confidence
    candidate_col = None
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, dict) and "label" in x and "confidence" in x).all():
            candidate_col = col
            break

    if not candidate_col:
        raise ValueError("No valid sentiment result column found.")

    df["sentiment_label"] = df[candidate_col].apply(lambda x: x["label"])
    df["confidence"] = df[candidate_col].apply(lambda x: x["confidence"])
    
    # إضافة الأعمدة التفصيلية لكل فئة (من score_distribution)
    df["score_negative"] = df[candidate_col].apply(lambda x: x.get("score_distribution", {}).get("negative", 0.0))
    df["score_neutral"] = df[candidate_col].apply(lambda x: x.get("score_distribution", {}).get("neutral", 0.0))
    df["score_positive"] = df[candidate_col].apply(lambda x: x.get("score_distribution", {}).get("positive", 0.0))

    return df

def plot_sentiment_distribution(df, output_dir="results"):
    counts = df["sentiment_label"].value_counts()
    counts = counts.reindex(["positive", "neutral", "negative"], fill_value=0)
    counts.plot(kind="bar", title="Sentiment Distribution", color=["green", "gray", "red"])
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sentiment_distribution.png")
    plt.close()

def generate_summary(df):
    summary = pd.DataFrame(columns=[
        "Total samples", "Positive %", "Neutral %", "Negative %",
        "High confidence max", "Low confidence min",
        "Low confidence count (< 0.6)", "Low conf. positive", "Low conf. neutral", "Low conf. negative"
    ])

    total = len(df)
    pos = (df["sentiment_label"] == "positive").sum()
    neu = (df["sentiment_label"] == "neutral").sum()
    neg = (df["sentiment_label"] == "negative").sum()

    high = df["confidence"].max()
    low = df["confidence"].min()

    low_df = df[df["confidence"] < 0.6]
    low_count = len(low_df)
    low_pos = (low_df["sentiment_label"] == "positive").sum()
    low_neu = (low_df["sentiment_label"] == "neutral").sum()
    low_neg = (low_df["sentiment_label"] == "negative").sum()

    summary.loc[0] = [
        total,
        round(pos / total * 100, 2),
        round(neu / total * 100, 2),
        round(neg / total * 100, 2),
        round(high, 4),
        round(low, 4),
        low_count,
        low_pos,
        low_neu,
        low_neg
    ]

    return summary
