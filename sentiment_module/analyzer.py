from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from tqdm import tqdm

class SentimentAnalyzer:
    def __init__(self):
        print("Loading sentiment analysis model...")
        self.tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        self.model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        self.labels = ["negative", "neutral", "positive"]

    def analyze_batch_json(self, json_data, batch_size=32):
        texts = json_data.get("texts", [])
        results = []

        print(f"Analyzing {len(texts)} texts in batches of {batch_size}...")

        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i + batch_size]
            batch_results = self._analyze_batch(batch_texts)
            results.extend(batch_results)

        return {"results": results}

    def _analyze_batch(self, texts):
        clean_texts = [t if isinstance(t, str) and t.strip() != "" else None for t in texts]
        encoded = self.tokenizer(
            [t if t else "" for t in clean_texts],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            logits = self.model(**encoded).logits
            probs = F.softmax(logits, dim=1).tolist()

        batch_results = []
        for i, text in enumerate(texts):
            if clean_texts[i] is None:
                batch_results.append({
                    "text": text,
                    "label": "undefined",
                    "confidence": 0.0,
                    "score_distribution": {
                        "negative": 0.0,
                        "neutral": 0.0,
                        "positive": 0.0
                    },
                    "error_message": "empty or invalid text"
                })
            else:
                pred_index = int(torch.tensor(probs[i]).argmax())
                batch_results.append({
                    "text": text,
                    "label": self.labels[pred_index],
                    "confidence": round(probs[i][pred_index], 4),
                    "score_distribution": {
                        "negative": round(probs[i][0], 4),
                        "neutral": round(probs[i][1], 4),
                        "positive": round(probs[i][2], 4)
                    },
                    "error_message": None
                })

        return batch_results
