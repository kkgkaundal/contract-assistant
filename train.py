from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments)
import evaluate, numpy as np, os, json

MODEL_NAME = "microsoft/deberta-v3-base"
LABELS = [
    "value_update", "clause_edit", "section_edit",
    "rename_party", "contract_q", "general_q", "chat"
]

# ---------------------------------------------------------------------
# 1. Load your data – expects ./data/train.jsonl & test.jsonl
#    [{"text": "Set amount to 5L", "label": "value_update"}, ...]
# ---------------------------------------------------------------------

def load_data():
    if not os.path.exists("data/train.jsonl"):
        raise SystemExit("✖  Put your labelled JSONL into ./data first » train.jsonl + test.jsonl")
    ds = load_dataset("json", data_files={
        "train": "data/train.jsonl",
        "test": "data/test.jsonl"
    })
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    ds = ds.class_encode_column("label")

    def tok(batch):
        return tokenizer(batch["text"], truncation=True)

    return ds.map(tok, batched=True), tokenizer

# ---------------------------------------------------------------------

def main():
    ds, tok = load_data()
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS))

    metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return metric.compute(predictions=preds, references=labels, average="micro")

    args = TrainingArguments(
        output_dir="checkpoints",
        learning_rate=2e-5,
        num_train_epochs=4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1"
    )

    Trainer(model=model, args=args, train_dataset=ds["train"],
            eval_dataset=ds["test"], tokenizer=tok,
            compute_metrics=compute_metrics).train()

    model.save_pretrained("model")
    tok.save_pretrained("model")
    print("✓ model saved → ./model/")

if __name__ == "__main__":
    main()