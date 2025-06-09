from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from .config import LABELS, MODEL_PATH, CONF_THRESHOLD

_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
_model.eval()

@torch.inference_mode()
def classify(text: str) -> tuple[str, float]:
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    logits = _model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze()
    conf_tensor, idx = probs.max(0)
    intent = _model.config.id2label[idx.item()]
    conf: float = float(conf_tensor)

    print(f"Probs: {probs}")
    print(f"Predicted intent: {intent} with confidence {conf}")

    if conf < CONF_THRESHOLD:
        intent = "fallback"
    return intent, conf
