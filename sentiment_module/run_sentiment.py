import json
import os
import pandas as pd
from datetime import datetime
from analyzer import SentimentAnalyzer
from utils import extract_sentiment_columns, plot_sentiment_distribution, generate_summary

# ============ 1. الإعدادات =============
INPUT_JSON_PATH = "input_texts.json"  # الباك إند رح يرسل هاد الملف
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_JSON_PATH = os.path.join(OUTPUT_DIR, "sentiment_results.json")
OUTPUT_CSV_PATH = os.path.join(OUTPUT_DIR, f"sentiment_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

# ============ 2. تحميل ملف JSON ============
print("Loading input JSON file...")
try:
    with open(INPUT_JSON_PATH, "r", encoding="utf-8") as f:
        input_data = json.load(f)
except Exception as e:
    print(f"Failed to load input JSON: {e}")
    exit(1)

texts = input_data.get("texts", [])
if not texts or not isinstance(texts, list):
    print("Invalid or empty 'texts' list in input JSON.")
    exit(1)

# ============ 3. تحليل المشاعر ============
print("Running sentiment analysis...")
analyzer = SentimentAnalyzer()
results_json = analyzer.analyze_batch_json({"texts": texts}, batch_size=32)
results = results_json["results"]

# ============ 4. حفظ النتائج كـ JSON ============
print(f"Saving JSON results to {OUTPUT_JSON_PATH} ...")
with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(results_json, f, ensure_ascii=False, indent=2)

# ============ 5. معالجة وتحليل النتائج ============
print("Processing DataFrame...")
df = pd.DataFrame()
df["sentiment_result"] = results
df = extract_sentiment_columns(df)

plot_sentiment_distribution(df, OUTPUT_DIR)
summary = generate_summary(df)

# ============ 6. حفظ التلخيص ============
summary.to_csv(OUTPUT_CSV_PATH, index=False)
print(f"Saved summary to {OUTPUT_CSV_PATH}")

# ============ 7. إنهاء ============
print("\nSummary:")
print(summary.to_string(index=False))
print("All done. JSON output is ready for backend.")
