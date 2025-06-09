# Contract Intent Classifier
Real‑time intent recognition for an e‑signing contract editor.  
– **WebSocket API** for lightning‑fast chat.  
– **DeBERTa‑v3 fine‑tuned** → ~99 % micro‑F1 on internal legal dataset.

## Quick start
```bash
# 1. install deps (inside virtual env)
pip install -r requirements.txt

# 2. train once (or copy your own model to ./model)
python train.py

# 3. run server
uvicorn app.main:app --reload