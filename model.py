"""model.py — logica del modello per il generatore di poesie.

Usa esclusivamente: stringhe, liste, tuple, dizionari, funzioni.
Nessuna classe, nessun ORM, nessuna struttura esterna.

Flusso principale:
  1. load_dataset()      → scarica le poesie dal dataset HuggingFace
  2. filter_by_theme()   → filtra per tema/tag
  3. sample_poems()      → seleziona un campione casuale come few-shot examples
  4. build_prompt()      → costruisce il prompt testuale per l'LLM
  5. call_llm()          → chiama HuggingFace Inference API (gratuita) e restituisce la poesia
"""

import random
import json
import urllib.request
import urllib.error


# ---------------------------------------------------------------------------
# Struttura dati del dataset
# ---------------------------------------------------------------------------
# Ogni poesia è un dizionario con le chiavi:
#   "title"  (str)  — titolo della poesia
#   "poem"   (str)  — testo completo della poesia
#   "author" (str)  — autore
#   "tags"   (str)  — tag separati da virgola (es. "love, nature, death")
#
# Il dataset globale è una lista di questi dizionari.

_dataset: list = []
_is_loaded: bool = False

# Modello HuggingFace gratuito da usare
HF_MODEL: str = "mistralai/Mistral-7B-Instruct-v0.3"


# ---------------------------------------------------------------------------
# 1. Caricamento dataset
# ---------------------------------------------------------------------------

def load_dataset(url: str = "") -> dict:
    """Scarica e carica il dataset di poesie in memoria.

    Usa la HuggingFace Datasets Viewer API (pubblica, no autenticazione).

    Returns:
        dict con chiavi: ok (bool), count (int), error (str|None)
    """
    global _dataset, _is_loaded

    api_url: str = (
        "https://datasets-server.huggingface.co/rows"
        "?dataset=suayptalha%2FPoetry-Foundation-Poems"
        "&config=default&split=train&offset=0&length=100"
    )

    try:
        req = urllib.request.Request(api_url, headers={"User-Agent": "poetry-gen/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw: str = resp.read().decode("utf-8")
        data: dict = json.loads(raw)
    except Exception as e:
        return {"ok": False, "count": 0, "error": str(e)}

    rows: list = data.get("rows", [])
    parsed: list = []
    for entry in rows:
        row: dict = entry.get("row", {})
        poem_dict: dict = {
            "title":  str(row.get("Title",  row.get("title",  ""))),
            "poem":   str(row.get("Poem",   row.get("poem",   ""))),
            "author": str(row.get("Poet",   row.get("author", ""))),
            "tags":   str(row.get("Tags",   row.get("tags",   ""))),
        }
        if poem_dict["poem"].strip():
            parsed.append(poem_dict)

    _dataset = parsed
    _is_loaded = True
    return {"ok": True, "count": len(_dataset), "error": None}


def is_loaded() -> bool:
    """Restituisce True se il dataset è stato caricato."""
    return _is_loaded and len(_dataset) > 0


def dataset_info() -> dict:
    """Restituisce statistiche sul dataset caricato."""
    if not _dataset:
        return {"count": 0, "themes": []}

    all_tags: list = []
    for poem in _dataset:
        tags_raw: str = poem.get("tags", "")
        for tag in tags_raw.split(","):
            tag = tag.strip().lower()
            if tag and tag not in all_tags:
                all_tags.append(tag)

    return {"count": len(_dataset), "themes": sorted(all_tags)}


# ---------------------------------------------------------------------------
# 2. Filtro per tema
# ---------------------------------------------------------------------------

def filter_by_theme(theme: str) -> list:
    """Restituisce la lista di poesie che contengono il tema nei tag o nel titolo."""
    if not theme or not _dataset:
        return list(_dataset)

    theme_lower: str = theme.lower().strip()
    filtered: list = []
    for poem in _dataset:
        tags_lower: str  = poem.get("tags", "").lower()
        title_lower: str = poem.get("title", "").lower()
        poem_lower: str  = poem.get("poem", "").lower()
        if theme_lower in tags_lower or theme_lower in title_lower or theme_lower in poem_lower:
            filtered.append(poem)

    return filtered if filtered else list(_dataset)


# ---------------------------------------------------------------------------
# 3. Campionamento few-shot
# ---------------------------------------------------------------------------

def sample_poems(poems: list, n: int = 2) -> tuple:
    """Seleziona n poesie casuali e le restituisce come tupla di dizionari."""
    n = min(n, len(poems))
    if n == 0:
        return ()
    sampled: list = random.sample(poems, n)
    result: list = []
    for p in sampled:
        result.append({
            "title":  p.get("title", ""),
            "author": p.get("author", ""),
            "poem":   p.get("poem", "")[:400],  # tronca per non superare i limiti del modello
        })
    return tuple(result)


# ---------------------------------------------------------------------------
# 4. Costruzione del prompt
# ---------------------------------------------------------------------------

def build_prompt(samples: tuple, theme: str, style: str, length: str) -> str:
    """Costruisce il prompt per HuggingFace Inference API (formato instruction).

    Mistral usa il formato [INST] ... [/INST] per le istruzioni.
    """
    length_map: dict = {
        "short":  "short (4-8 lines)",
        "medium": "medium length (10-16 lines)",
        "long":   "long (20+ lines)",
    }
    length_desc: str = length_map.get(length, "medium length (10-16 lines)")

    # esempi few-shot come stringa
    examples_parts: list = []
    for i, sample in enumerate(samples):
        block: str = (
            f"Example {i+1}:\n"
            f"Title: {sample['title']} by {sample['author']}\n"
            f"{sample['poem']}\n"
        )
        examples_parts.append(block)
    examples_str: str = "\n".join(examples_parts)

    # prompt nel formato istruzione di Mistral
    prompt: str = (
        f"[INST] You are a talented poet. "
        f"Write an original poem inspired by the following examples.\n\n"
        f"{examples_str}\n"
        f"Now write a NEW original poem with these requirements:\n"
        f"- Theme: {theme if theme else 'free'}\n"
        f"- Style: {style if style else 'free verse'}\n"
        f"- Length: {length_desc}\n"
        f"Write ONLY the poem, no explanations, no title prefix, no introductory sentences. "
        f"Start directly with the first line. [/INST]"
    )
    return prompt


# ---------------------------------------------------------------------------
# 5. Chiamata all'LLM (HuggingFace Inference API — gratuita)
# ---------------------------------------------------------------------------

def call_llm(prompt: str, api_key: str) -> dict:
    """Chiama HuggingFace Inference API con il modello Mistral (gratuito).

    Endpoint: https://api-inference.huggingface.co/models/<model>

    Args:
        prompt:  stringa prompt costruita da build_prompt()
        api_key: HuggingFace token (inizia con hf_...)

    Returns:
        dict con chiavi: ok (bool), poem (str), error (str|None)
    """
    endpoint: str = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

    body: dict = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens":   400,
            "temperature":      0.85,
            "top_p":            0.92,
            "do_sample":        True,
            "return_full_text": False,   # restituisce SOLO il testo generato, non il prompt
        }
    }

    body_bytes: bytes = json.dumps(body).encode("utf-8")

    headers: dict = {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    try:
        req = urllib.request.Request(endpoint, data=body_bytes, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw: str = resp.read().decode("utf-8")
        response_data = json.loads(raw)
    except urllib.error.HTTPError as e:
        error_body: str = e.read().decode("utf-8")
        # errore 503 = modello in avvio (warm-up), riprova dopo qualche secondo
        if e.code == 503:
            return {"ok": False, "poem": "", "error": "Il modello si sta avviando su HuggingFace (può richiedere 20-30 secondi). Riprova tra poco."}
        return {"ok": False, "poem": "", "error": f"HTTP {e.code}: {error_body}"}
    except Exception as e:
        return {"ok": False, "poem": "", "error": str(e)}

    # la risposta è una lista: [{"generated_text": "..."}]
    poem_text: str = ""
    if isinstance(response_data, list) and len(response_data) > 0:
        poem_text = response_data[0].get("generated_text", "")
    elif isinstance(response_data, dict):
        poem_text = response_data.get("generated_text", "")

    if not poem_text:
        return {"ok": False, "poem": "", "error": "Risposta vuota dal modello. Riprova."}

    return {"ok": True, "poem": poem_text.strip(), "error": None}


# ---------------------------------------------------------------------------
# 6. Pipeline completa
# ---------------------------------------------------------------------------

def generate_poem(theme: str, style: str, length: str, api_key: str) -> dict:
    """Pipeline completa: filtra → campiona → costruisce prompt → chiama LLM.

    Returns:
        dict con: ok (bool), poem (str), samples_used (int), error (str|None)
    """
    if not is_loaded():
        return {"ok": False, "poem": "", "samples_used": 0, "error": "Dataset non caricato. Clicca 'Carica Dataset' prima."}

    filtered: list  = filter_by_theme(theme)
    samples:  tuple = sample_poems(filtered, n=2)
    prompt:   str   = build_prompt(samples, theme, style, length)
    result:   dict  = call_llm(prompt, api_key)

    result["samples_used"] = len(samples)
    return result