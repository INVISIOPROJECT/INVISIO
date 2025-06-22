from datasets import DatasetDict, Dataset
import pandas as pd

# دالة لتحويل ملف BIO إلى قائمة جمل، كل جملة عبارة عن dict {"tokens": [], "ner_tags": []}
def load_bio_file(filename, label2id):
    with open(filename, encoding="utf-8") as f:
        tokens = []
        labels = []
        data = []

        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    data.append({"tokens": tokens, "ner_tags": [label2id[l] for l in labels]})
                    tokens, labels = [], []
            else:
                word, tag = line.split()
                tokens.append(word)
                labels.append(tag)

        if tokens:
            data.append({"tokens": tokens, "ner_tags": [label2id[l] for l in labels]})

    return data

# تعريف جميع التاغات المستخدمة
unique_labels = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-PRODUCT', 'I-PRODUCT', 'B-EVENT', 'I-EVENT']
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}

# تحميل البيانات من الملفات
train_data = load_bio_file("train.txt", label2id)
dev_data = load_bio_file("dev.txt", label2id)
test_data = load_bio_file("test.txt", label2id)

# إنشاء Hugging Face Dataset
datasets = DatasetDict({
    "train": Dataset.from_list(train_data),
    "validation": Dataset.from_list(dev_data),
    "test": Dataset.from_list(test_data),
})

# عرض مثال
datasets["train"][0]

from transformers import AutoTokenizer, AutoModelForTokenClassification

model_checkpoint = "Davlan/bert-base-multilingual-cased-ner-hrl"

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Load model with NEW classification layer (ignore old head)
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True 

)
from transformers import DataCollatorForTokenClassification
from transformers import PreTrainedTokenizerBase
from typing import List, Tuple

# تاغ للإشارة إلى الكلمات التي لا يجب حسابها أثناء التدريب (مثل التوكنات الفرعية)
IGNORE_INDEX = -100

# دالة لتوسيع التاغات بحيث تتوافق مع التوكنات بعد التقسيم
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding=True,
    )

    labels = []
    for i in range(len(examples["tokens"])):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(IGNORE_INDEX)
            elif word_idx != previous_word_idx:
                label_ids.append(examples["ner_tags"][i][word_idx])
            else:
                label_ids.append(IGNORE_INDEX)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# تطبيقها على كل الـ datasets
tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)

# إنشاء data collator ذكي لدمج البيانات أثناء التدريب
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
import evaluate

# التقييم باستخدام مكتبة seqeval (دقة وتذكّر وF1)
seqeval = evaluate.load("seqeval")

# دالة لحساب الميتريك
def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.argmax(axis=-1)

    true_labels = [
        [id2label[label] for (pred, label) in zip(prediction, label_list) if label != IGNORE_INDEX]
        for prediction, label_list in zip(predictions, labels)
    ]
    true_predictions = [
        [id2label[pred] for (pred, label) in zip(prediction, label_list) if label != IGNORE_INDEX]
        for prediction, label_list in zip(predictions, labels)
    ]

    return seqeval.compute(predictions=true_predictions, references=true_labels)

# إعدادات التدريب
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
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_overall_f1",
    push_to_hub=False
)

# إنشاء المدرب
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)
trainer.train()
trainer.save_model("ner_trained_model2")
tokenizer.save_pretrained("ner_trained_model2")
