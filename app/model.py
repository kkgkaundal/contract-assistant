from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from .config import LABELS, MODEL_PATH, CONF_THRESHOLD

_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
_model.eval()

@torch.inference_mode()
def classify(text: str):
    enc = _tokenizer(text, return_tensors="pt", truncation=True)
    logits = _model(**enc).logits
    probs = torch.softmax(logits, dim=-1).squeeze()
    conf, idx = probs.max(0)
    intent = LABELS[idx]
    conf = float(conf)
    if conf < CONF_THRESHOLD:
        intent = "fallback"
    return intent, conf