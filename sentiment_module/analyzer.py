# analyzer.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

class SentimentAnalyzer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        self.model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        self.labels = ["negative", "neutral", "positive"]

    def analyze_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = F.softmax(logits, dim=1).squeeze().tolist()
        pred_index = int(torch.argmax(torch.tensor(probs)))
        return {
            "text": text,
            "label": self.labels[pred_index],
            "confidence": round(probs[pred_index], 4),
            "score_distribution": {
                "negative": round(probs[0], 4),
                "neutral": round(probs[1], 4),
                "positive": round(probs[2], 4)
            }
        }

    def analyze_batch(self, texts):
        results = []
        valid_texts = [t if isinstance(t, str) and t.strip() != "" else "" for t in texts]

        for text in valid_texts:
            if text == "":
                results.append({
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
                try:
                    result = self.analyze_text(text)
                    result["error_message"] = None
                    results.append(result)
                except Exception as e:
                    results.append({
                        "text": text,
                        "label": "error",
                        "confidence": 0.0,
                        "score_distribution": {
                            "negative": 0.0,
                            "neutral": 0.0,
                            "positive": 0.0
                        },
                        "error_message": str(e)
                    })
        return results
