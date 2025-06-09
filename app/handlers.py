"""All domain‑specific actions happen here."""

def handle_value_update(text: str):
    # parse numbers / fields, then update DB…
    return {"status": "ok", "action": "value_update", "msg": "Amount updated"}

def handle_clause_edit(text):
    return {"status": "ok", "action": "clause_edit"}

def handle_section_edit(text):
    return {"status": "ok", "action": "section_edit"}

def handle_party_rename(text):
    return {"status": "ok", "action": "rename_party"}

def handle_contract_question(text):
    return {"answer": "Clause 7 = …"}

def handle_general_question(text):
    return {"answer": "GST return is due on…"}

def handle_chat(text):
    return {"reply": "Glad to help!"}

def handle_fallback(text):
    return {"error": "I'm not sure – could you rephrase?"}

ROUTER = {
    "value_update": handle_value_update,
    "clause_edit": handle_clause_edit,
    "section_edit": handle_section_edit,
    "rename_party": handle_party_rename,
    "contract_q": handle_contract_question,
    "general_q": handle_general_question,
    "chat": handle_chat,
    "fallback": handle_fallback,
}

def route(intent, text):
    return ROUTER[intent](text)