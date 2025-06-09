from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from app.config import MODEL_PATH, CONF_THRESHOLD

_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
_model.eval()

@torch.inference_mode()
def classify(text: str) -> tuple[str, float]:
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    logits = _model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze()
    conf_tensor, idx = probs.max(0)
    intent = _model.config.id2label[idx.item()]  # ✅ FIXED
    conf: float = float(conf_tensor)

    if conf < CONF_THRESHOLD:
        intent = "fallback"
    return intent, conf

# Example usage
if __name__ == "__main__":
    test_text = "Please update clause 3 to reflect the new policy"
    intent, confidence = classify(test_text)
    print(f"Predicted Intent: {intent} ({confidence:.2f})")
