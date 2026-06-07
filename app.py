"""app.py — server FastAPI per il generatore di poesie LLM.

Struttura delle rotte (speculare al progetto della professoressa):
    GET  /       → frontend HTML
    POST /train  → carica il dataset di poesie da HuggingFace
    POST /generate → genera una poesia con l'LLM

Stato globale:
    app.state.api_key  — chiave API Anthropic (inserita dall'utente via /train)
    Il dataset è gestito in model.py con variabili di modulo.
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

import model  # importa tutte le funzioni di model.py


app = FastAPI(title="Poetry Generator")

# monta static e templates (stessa struttura del progetto della prof)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# workaround per url_for('static', filename=...) nei template
def _template_url_for(name: str, **path_params):
    if name == "static" and "filename" in path_params:
        path_params["path"] = path_params.pop("filename")
    return app.url_path_for(name, **path_params)

templates.env.globals["url_for"] = _template_url_for

# stato dell'applicazione: api_key come stringa
app.state.api_key: str = ""


# ---------------------------------------------------------------------------
# GET / — serve il frontend
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "title": "Poetry Generator"})


# ---------------------------------------------------------------------------
# POST /train — carica il dataset e memorizza la chiave API
# ---------------------------------------------------------------------------

@app.post("/train")
async def train(request: Request):
    """Carica il dataset di poesie da HuggingFace.

    Body JSON atteso:
        { "api_key": "<chiave-anthropic>", "url": "<opzionale>" }

    La chiave API viene salvata in app.state.api_key per le chiamate /generate.
    Il campo "url" è opzionale: se omesso si usa il dataset di default.

    Response JSON di successo:
        { "ok": true, "count": <int>, "themes": [...] }
    """
    data: dict = await request.json()

    # salva la chiave API (stringa)
    api_key: str = data.get("api_key", "").strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="Campo 'api_key' obbligatorio.")
    app.state.api_key = api_key

    # carica il dataset (url opzionale)
    url: str = data.get("url", "")
    if url:
        result: dict = model.load_dataset(url)
    else:
        result: dict = model.load_dataset()

    if not result["ok"]:
        raise HTTPException(status_code=400, detail=result["error"])

    info: dict = model.dataset_info()
    return JSONResponse({
        "ok":     True,
        "count":  result["count"],
        "themes": info["themes"][:30],   # restituisce max 30 temi come lista
    })


# ---------------------------------------------------------------------------
# POST /generate — genera una poesia
# ---------------------------------------------------------------------------

@app.post("/generate")
async def generate(request: Request):
    """Genera una poesia usando l'LLM.

    Body JSON (tutti i campi opzionali):
        {
          "theme":  "nature",      // tema della poesia
          "style":  "sonnet",      // stile (verso libero, sonetto, haiku, ...)
          "length": "medium"       // "short" | "medium" | "long"
        }

    Response JSON:
        { "ok": true, "poem": "...", "samples_used": 3 }
    """
    if not app.state.api_key:
        raise HTTPException(status_code=400, detail="API key non impostata. Chiama /train prima.")

    if not model.is_loaded():
        raise HTTPException(status_code=400, detail="Dataset non caricato. Chiama /train prima.")

    data: dict = await request.json()
    theme:  str = data.get("theme",  "")
    style:  str = data.get("style",  "verso libero")
    length: str = data.get("length", "medium")

    if length not in ("short", "medium", "long"):
        raise HTTPException(status_code=400, detail="'length' deve essere 'short', 'medium' o 'long'.")

    result: dict = model.generate_poem(theme, style, length, app.state.api_key)

    if not result["ok"]:
        raise HTTPException(status_code=400, detail=result["error"])

    return JSONResponse({
        "ok":          True,
        "poem":        result["poem"],
        "samples_used": result["samples_used"],
    })


# ---------------------------------------------------------------------------
# GET /status — info sullo stato corrente (utile per debug)
# ---------------------------------------------------------------------------

@app.get("/status")
async def status():
    """Restituisce lo stato corrente del sistema."""
    info: dict = model.dataset_info()
    return JSONResponse({
        "dataset_loaded": model.is_loaded(),
        "poem_count":     info["count"],
        "api_key_set":    bool(app.state.api_key),
        "themes_sample":  info["themes"][:10],
    })


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
