from datasets import load_dataset, ClassLabel
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments)
import evaluate, numpy as np, os
from app.config import LABELS, MODEL_PATH

MODEL_NAME = "microsoft/deberta-v3-base"

def load_data():
    if not os.path.exists("data/train.jsonl"):
        raise SystemExit("✖ Missing data/train.jsonl and test.jsonl")

    ds = load_dataset("json", data_files={
        "train": "data/train.jsonl",
        "test":  "data/test.jsonl"
    })

    ds = ds.cast_column("label", ClassLabel(names=LABELS))
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tok(batch["text"], truncation=True, max_length=5000)
    ds = ds.map(tokenize, batched=True)
    return ds, tok

def main():
    ds, tok = load_data()

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label={i:l for i,l in enumerate(LABELS)},
        label2id={l:i for i,l in enumerate(LABELS)}
    )

    f1 = evaluate.load("f1")
    def metrics(p):
        preds = np.argmax(p.predictions, axis=-1)
        return f1.compute(predictions=preds, references=p.label_ids,
                          average="micro")

    args = TrainingArguments(
        output_dir="checkpoints",
        save_strategy="no",               # ← key change
        evaluation_strategy="epoch",
        num_train_epochs=5,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        weight_decay=0.01,
        fp16=True,                        # use mixed-precision
        logging_dir="./logs",
        report_to="none"
    )

    trainer = Trainer(model=model, args=args,
                      train_dataset=ds["train"],
                      eval_dataset=ds["test"],
                      tokenizer=tok,
                      compute_metrics=metrics)

    trainer.train()

    # single final save
    trainer.save_model(MODEL_PATH)
    tok.save_pretrained(MODEL_PATH)
    print("✓ Model saved at", MODEL_PATH)

if __name__ == "__main__":
    main()
