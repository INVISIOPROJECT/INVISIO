# run_sentiment.py

import pandas as pd
import json
from analyzer import SentimentAnalyzer
from utils import extract_sentiment_columns, plot_sentiment_distribution, generate_summary
from datetime import datetime

# 1. قراءة البيانات
CSV_PATH = "news_data.csv"  # عدل المسار إذا لزم
TEXT_COLUMN = "Headlines"        # عدل العمود حسب الحاجة
SAVE_JSON_PATH = "sentiment_results.json"
SAVE_JSON_PATH = "sentiment_results.json"

print(" Loading data...")
df = pd.read_csv(CSV_PATH)
texts = df[TEXT_COLUMN].fillna("").tolist()

# 2. تحليل المشاعر
print(" Running sentiment analysis...")
sa = SentimentAnalyzer()
results = sa.analyze_batch(texts)

# 3. حفظ النتائج إلى JSON
print(" Saving JSON results...")
with open(SAVE_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# 4. دمج النتائج ضمن الداتا فريم
print(" Processing DataFrame...")
df["sentiment_result"] = results
df = extract_sentiment_columns(df)

# 5. عرض رسم بياني
plot_sentiment_distribution(df)

# 6. عرض ملخص
summary = generate_summary(df)
print("\n Summary:")
print(summary.to_string(index=False))

# 7. حفظ الملخص كـ CSV
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
summary.to_csv(f"sentiment_summary_{timestamp}.csv", index=False)
print("All done.")