Poetry Generator — LM demo con LLM Anthropic
=============================================

Struttura del progetto
-----------------------
```
poetry-gen/
├── app.py              — server FastAPI (rotte: /, /train, /generate, /status)
├── model.py            — logica del modello (solo funzioni, nessuna classe)
├── requirements.txt    — dipendenze Python
├── static/
│   └── main.css        — stile
└── templates/
    └── index.html      — frontend single-page (HTML + vanilla JS)
```

Avvio rapido
-------------
```bash
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
# oppure
uvicorn app:app --reload --host 0.0.0.0 --port 5000
```

Apri http://localhost:5000 nel browser.

Strutture dati usate (solo primitive Python)
---------------------------------------------
- str    → prompt, testo generato, chiave API
- list   → lista di dizionari poesia dal dataset
- tuple  → campioni few-shot (immutabile)
- dict   → singola poesia {"title","poem","author","tags"}, risposta JSON

Flusso
-------
1. Utente inserisce la chiave API e preme "Carica Dataset"
   → POST /train → model.load_dataset() scarica 100 poesie da HuggingFace
2. Utente sceglie tema, stile, lunghezza e preme "Genera"
   → POST /generate → model.generate_poem():
       a. filter_by_theme()  — filtra poesie per tema (list)
       b. sample_poems()     — sceglie 3 esempi casuali (tuple)
       c. build_prompt()     — costruisce il prompt (str)
       d. call_llm()         — chiama l'API Anthropic (dict risposta)

API
----
POST /train    body: { "api_key": "sk-ant-..." }
               resp: { "ok": true, "count": 100, "themes": [...] }

POST /generate body: { "theme": "nature", "style": "sonnet", "length": "medium" }
               resp: { "ok": true, "poem": "...", "samples_used": 3 }

GET  /status   resp: { "dataset_loaded": true, "poem_count": 100, ... }
