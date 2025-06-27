from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, EarlyStoppingCallback, DataCollatorForTokenClassification
from datasets import Dataset, DatasetDict
from sklearn.metrics import classification_report
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from itertools import chain
import json
import shutil
import evaluate
import torch


def read_bio_file_to_df(path):
    sentences, tags = [], []
    with open(path, 'r', encoding='utf-8') as f:
        tokens, ner_tags = [], []
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sentences.append(tokens)
                    tags.append(ner_tags)
                    tokens, ner_tags = [], []
                continue
            try:
                token, tag = line.split()
                tokens.append(token)
                ner_tags.append(tag)
            except ValueError:
                continue
        if tokens:
            sentences.append(tokens)
            tags.append(ner_tags)
    return pd.DataFrame({'tokens': sentences, 'tags': tags})


train_df = read_bio_file_to_df("newtrain (2).txt")
dev_df = read_bio_file_to_df("newdev.txt")
test_df = read_bio_file_to_df("newtest.txt")


label2id = {
    'B-EVENT': 0, 'B-LOC': 1, 'B-ORG': 2, 'B-PER': 3,
    'I-EVENT': 4, 'I-LOC': 5, 'I-ORG': 6, 'I-PER': 7,
    'O': 8
}
id2label = {v: k for k, v in label2id.items()}
IGNORE_INDEX = -100


model_checkpoint = "Jean-Baptiste/roberta-large-ner-english"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=64
    )
    labels = []
    for i, label in enumerate(examples["tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(IGNORE_INDEX)
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[label[word_idx]])
            else:
                label_ids.append(label2id[label[word_idx]])
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df),
    "validation": Dataset.from_pandas(dev_df),
    "test": Dataset.from_pandas(test_df)
})
tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)


try:
    seqeval = evaluate.load("seqeval")
except:
    seqeval = None


def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.argmax(axis=-1)
    if seqeval is None:
        return {}
    true_labels = [
        [id2label[label] for (pred, label) in zip(prediction, label_list) if label != IGNORE_INDEX]
        for prediction, label_list in zip(predictions, labels)
    ]
    true_predictions = [
        [id2label[pred] for (pred, label) in zip(prediction, label_list) if label != IGNORE_INDEX]
        for prediction, label_list in zip(predictions, labels)
    ]
    return seqeval.compute(predictions=true_predictions, references=true_labels)


training_args = TrainingArguments(
    output_dir="./ner-model",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir="./logs",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_overall_f1",
    push_to_hub=False
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

trainer.train()


trainer.save_model("ner_trained_model_final")
tokenizer.save_pretrained("ner_trained_model_final")
shutil.make_archive("ner_trained_model_final", 'zip', "ner_trained_model_final")

metrics = trainer.evaluate(tokenized_datasets["validation"])
with open("training_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

trainer.evaluate(tokenized_datasets["test"])


sentences = list(zip(train_df["tokens"], train_df["tags"]))
model.eval()
predicted_labels = []
sent_texts = [" ".join(tokens) for tokens, _ in sentences]
true_labels = [tags for _, tags in sentences]

for text in sent_texts:
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs).logits
    preds = torch.argmax(outputs, dim=-1).squeeze().tolist()
    word_ids = inputs.word_ids(batch_index=0)
    tags, prev_word = [], None
    for i, word_id in enumerate(word_ids):
        if word_id is None or word_id == prev_word:
            continue
        label = model.config.id2label[preds[i]]
        tags.append(label)
        prev_word = word_id
    predicted_labels.append(tags)


rows = []
for sent, true, pred in zip(sentences, true_labels, predicted_labels):
    for token, gold, pred_tag in zip(sent, true, pred):
        rows.append({"Token": token, "True": gold, "Predicted": pred_tag})
    rows.append({})

df_out = pd.DataFrame(rows)
df_out.to_csv("ner_finetuned_predictions.csv", index=False)
df_out = df_out.dropna(subset=["Token"])
y_true = df_out["True"].tolist()
y_pred = df_out["Predicted"].tolist()
report = classification_report(y_true, y_pred, digits=4)
print("\n📊 Classification Report:\n")
print(report)
