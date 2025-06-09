from fastapi import FastAPI, WebSocket, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from .model import classify
from .handlers import route

app = FastAPI(title="Contract Intent WebSocket API")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    while True:
        try:
            data = await ws.receive_text()
        except Exception:
            break
        intent, conf = classify(data)
        payload = route(intent, data)
        payload.update({"intent": intent, "confidence": conf})
        await ws.send_json(payload)