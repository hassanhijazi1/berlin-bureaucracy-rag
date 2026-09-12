"""
FastAPI service for the Berlin bureaucracy RAG.

Run:
    uvicorn api:app --reload --port 8000

Open:
    http://127.0.0.1:8000
    http://127.0.0.1:8000/docs
    http://127.0.0.1:8000/health

The corpus is built once at startup, not per request.
"""

import re
import time
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from rag import System, K


# ================================================================
# CONFIG
# ================================================================

MAX_QUESTION_CHARS = 500
DAILY_REQUEST_LIMIT = 200


# ================================================================
# CITATION HELPERS
# ================================================================

CITE_RE = re.compile(
    r"\s*\[[A-Za-z0-9_ äöüÄÖÜß.-]+?\.(?:md|txt)"
    r"(?:\s*[;,]\s*[A-Za-z0-9_ äöüÄÖÜß.-]+?\.(?:md|txt))*\]",
    re.IGNORECASE,
)


PAGE_NAMES = {
    "anmeldung_hauptwohnung.md":
        "Berlin.de — Anmeldung einer Wohnung",

    "anmeldung_hauptwohnung_leicht.md":
        "Berlin.de — Anmeldung (Leichte Sprache)",

    "anmeldung_nebenwohnung.md":
        "Berlin.de — Anmeldung einer Nebenwohnung",

    "lea_studium.md":
        "LEA — Aufenthaltserlaubnis zum Studium",

    "lea_aufenthaltserlaubnis_studium.md":
        "LEA — Studium (Merkblatt)",

    "lea_krankenversicherung.md":
        "LEA — Krankenversicherung",

    "lea_termine.md":
        "LEA — Termine",

    "lea_erloeschen.md":
        "LEA — Erlöschen des Aufenthaltstitels",

    "lea_visum_national.md":
        "LEA — Einreise mit nationalem Visum",

    "lea_kurzaufenthalt_90tage.md":
        "LEA — Kurzaufenthalt bis 90 Tage",

    "visum_studium.md":
        "Auswärtiges Amt — Visum zum Studium",

    "gkv_versicherte.md":
        "Bundesgesundheitsministerium — GKV",

    "tk_studenten_ausland.md":
        "TK — Studierende aus dem Ausland",

    "krankenkasse_studium.md":
        "TK — Krankenversicherung für Studierende",
}


LAWS = {
    "bmg": "BMG",
    "aufenthg": "AufenthG",
    "aufenthv": "AufenthV",
    "sgb": "SGB V",
}


def readable(source: str) -> str:
    """Turn a filename into a human-readable source label."""

    if source in PAGE_NAMES:
        return PAGE_NAMES[source]

    stem = (
        source
        .replace(".txt", "")
        .replace(".md", "")
    )

    parts = stem.split("_")

    # Example:
    # bmg_de_17 -> BMG § 17
    # aufenthg_en_16b -> AufenthG § 16b

    if parts[0] in LAWS and len(parts) >= 3:
        return (
            f"{LAWS[parts[0]]} § {parts[2]}"
        )

    # Example:
    # vab_a16b -> VAB A 16b
    if parts[0] == "vab" and len(parts) >= 2:
        return (
            f"VAB A {parts[1].lstrip('a')} "
            f"— LEA-Verfahrenshinweise"
        )

    return source


def cited_files(text: str):
    """Extract cited filenames from [file.txt] markers."""

    citations = re.findall(
        r"\[([^\]]+)\]",
        text,
    )

    files = set()

    for citation in citations:
        for item in re.split(
            r"[;,]",
            citation,
        ):
            item = item.strip()

            if re.search(
                r"\.(?:txt|md)$",
                item,
                re.IGNORECASE,
            ):
                files.add(item)

    return files


def strip_citations(text: str) -> str:
    """
    Remove inline [filename] markers because sources are shown
    separately in the frontend.
    """

    text = CITE_RE.sub("", text)

    # Remove spaces before punctuation.
    text = re.sub(
        r"\s+([.,;:!?])",
        r"\1",
        text,
    )

    # Collapse excessive blank lines.
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


# ================================================================
# APP
# ================================================================

app = FastAPI(
    title="Berlin bureaucracy RAG",
    description=(
        "Answers questions about German registration and residence law "
        "for international students in Berlin, from cited sources. "
        "Declines when the sources do not cover the question. "
        "Not legal advice."
    ),
    version="1.0",
)


# ================================================================
# SYSTEM
# ================================================================

# Built ONCE when the FastAPI process starts.
system = System()

state = {
    "count": 0,
    "started": time.time(),
}


# ================================================================
# MODELS
# ================================================================

class Question(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=MAX_QUESTION_CHARS,
        examples=[
            "How long do I have to register my address?"
        ],
    )


class Source(BaseModel):
    file: str
    label: str


class Answer(BaseModel):
    answer: str
    refused: bool
    sources: List[Source]
    elapsed_seconds: float


# ================================================================
# ENDPOINT: ASK
# ================================================================

@app.post(
    "/ask",
    response_model=Answer,
)
def ask(q: Question) -> Answer:
    """Answer one question from the corpus."""

    question = q.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="The question is empty.",
        )

    if state["count"] >= DAILY_REQUEST_LIMIT:
        raise HTTPException(
            status_code=429,
            detail="Request limit reached for this instance.",
        )

    state["count"] += 1

    started = time.time()

    try:
        text, hits = system.ask(question)

    except Exception as exc:
        # IMPORTANT:
        # Print the complete underlying error to the terminal.
        # This makes BadRequestError debugging much easier.
        print()
        print("=" * 70)
        print("MODEL ERROR")
        print("=" * 70)
        print(
            f"{type(exc).__name__}: {exc}"
        )
        print("=" * 70)
        print()

        raise HTTPException(
            status_code=502,
            detail=(
                f"The model call failed: "
                f"{type(exc).__name__}"
            ),
        )

    if not text or not text.strip():
        text = (
            "The sources I have do not cover this."
        )

    refused = (
        "do not cover"
        in text.lower()
    )

    cited_sources = cited_files(text)

    files = sorted(cited_sources)

    return Answer(
        answer=strip_citations(text),
        refused=refused,
        sources=(
            []
            if refused
            else [
                Source(
                    file=f,
                    label=readable(f),
                )
                for f in files
            ]
        ),
        elapsed_seconds=round(
            time.time() - started,
            2,
        ),
    )


# ================================================================
# ENDPOINT: HEALTH
# ================================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "documents": len(system.documents),
        "chunks": len(system.chunks),
        "k": K,
        "requests_served": state["count"],
        "uptime_seconds": round(
            time.time() - state["started"]
        ),
    }


# ================================================================
# ENDPOINT: HOME
# ================================================================

@app.get(
    "/",
    response_class=HTMLResponse,
)
def home():
    return PAGE


# ================================================================
# FRONTEND
# ================================================================

PAGE = """<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Berlin registration and residence questions</title>

<style>
:root {
  --ink:   #16161a;
  --body:  #3d3d42;
  --muted: #8a8a90;
  --line:  #e8e8ea;
  --bg:    #fbfbfc;
  --accent:#1a4d8f;
}

* {
  box-sizing: border-box;
}

body {
  font-family:
    -apple-system,
    BlinkMacSystemFont,
    "Segoe UI",
    system-ui,
    sans-serif;

  background: var(--bg);
  color: var(--body);

  max-width: 680px;
  margin: 0 auto;

  padding: 64px 24px 96px;

  line-height: 1.65;
  font-size: 16px;

  -webkit-font-smoothing: antialiased;
}

h1 {
  font-size: 24px;
  font-weight: 650;
  letter-spacing: -0.01em;
  color: var(--ink);
  margin: 0 0 10px;
}

.lede {
  font-size: 14.5px;
  color: var(--muted);
  margin: 0 0 6px;
}

.disclaimer {
  font-size: 13px;
  color: var(--muted);
  margin: 0 0 32px;
  padding-left: 12px;
  border-left: 2px solid var(--line);
}

.disclaimer b {
  color: var(--body);
  font-weight: 600;
}

form {
  display: flex;
  gap: 10px;
  align-items: stretch;
}

input {
  flex: 1;
  padding: 13px 16px;
  font-size: 16px;
  font-family: inherit;
  color: var(--ink);
  background: #fff;
  border: 1px solid #d8d8dc;
  border-radius: 10px;
  transition:
    border-color .15s,
    box-shadow .15s;
}

input::placeholder {
  color: #b4b4ba;
}

input:focus {
  outline: none;
  border-color: var(--accent);
  box-shadow:
    0 0 0 3px rgba(26,77,143,.10);
}

button {
  padding: 0 22px;
  font-size: 15px;
  font-weight: 550;
  font-family: inherit;
  color: #fff;
  background: var(--ink);
  border: 0;
  border-radius: 10px;
  cursor: pointer;
  transition: background .15s;
}

button:hover:not(:disabled) {
  background: #000;
}

button:disabled {
  background: #c4c4c8;
  cursor: default;
}

.examples {
  margin-top: 14px;
  font-size: 13.5px;
  color: var(--muted);
}

.examples button {
  all: unset;
  cursor: pointer;
  color: var(--accent);
  border-bottom:
    1px solid rgba(26,77,143,.25);
  padding-bottom: 1px;
}

.examples button:hover {
  border-bottom-color: var(--accent);
}

.examples .sep {
  margin: 0 8px;
  color: #d0d0d4;
}

#out {
  margin-top: 40px;
}

#out:empty {
  display: none;
}

.card {
  background: #fff;
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 24px 26px;
  box-shadow:
    0 1px 2px rgba(0,0,0,.03);
}

.answer {
  color: var(--body);
}

.answer p {
  margin: 0 0 14px;
}

.answer p:last-child {
  margin-bottom: 0;
}

.answer b,
.answer strong {
  color: var(--ink);
  font-weight: 600;
}

.answer ul {
  margin: 0 0 14px;
  padding-left: 22px;
}

.answer li {
  margin-bottom: 7px;
}

.answer li:last-child {
  margin-bottom: 0;
}

.refused {
  color: var(--muted);
  font-style: italic;
}

.sources {
  margin-top: 22px;
  padding-top: 16px;
  border-top: 1px solid var(--line);
  font-size: 13px;
  color: var(--muted);
}

.sources .label {
  display: block;
  font-size: 11px;
  letter-spacing: .07em;
  text-transform: uppercase;
  color: #a8a8ae;
  margin-bottom: 8px;
}

.sources ul {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.sources li {
  background: #f4f4f6;
  border-radius: 6px;
  padding: 3px 9px;
  font-size: 12.5px;
  color: var(--body);
}

.timing {
  margin-top: 12px;
  font-size: 12px;
  color: #c0c0c6;
}

.thinking {
  color: var(--muted);
  font-size: 14.5px;
}

.thinking::after {
  content: '';
  display: inline-block;
  width: 1em;
  text-align: left;
  animation:
    dots 1.2s steps(4, end) infinite;
}

@keyframes dots {
  0%   { content: ''; }
  25%  { content: '.'; }
  50%  { content: '..'; }
  75%  { content: '...'; }
}

.error {
  background: #fdf3f3;
  border: 1px solid #f2d5d5;
  border-radius: 10px;
  padding: 14px 18px;
  color: #9b3b3b;
  font-size: 14.5px;
}

@media (max-width: 560px) {
  body {
    padding: 40px 18px 64px;
  }

  form {
    flex-direction: column;
  }

  button {
    padding: 13px;
  }
}
</style>

<h1>Berlin registration and residence questions</h1>

<p class="lede">
Answers come from the Bundesmeldegesetz, Aufenthaltsgesetz,
Aufenthaltsverordnung, SGB&nbsp;V, the Berlin LEA guidance and Berlin service
pages — and nothing else.
</p>

<p class="disclaimer">
<b>Not legal advice.</b>
The system declines when its sources do not cover a question.
Check the cited section before acting; annual figures may be out of date.
</p>

<form onsubmit="go(event)">
  <input
    id="q"
    maxlength="500"
    placeholder="How long do I have to register my address?"
    autofocus
  >
  <button id="btn" type="submit">
    Ask
  </button>
</form>

<div class="examples">

  <button
    type="button"
    onclick="fill(this)"
  >
    What happens if I register late?
  </button>

  <span class="sep">·</span>

  <button
    type="button"
    onclick="fill(this)"
  >
    How many days can I work as a student?
  </button>

  <span class="sep">·</span>

  <button
    type="button"
    onclick="fill(this)"
  >
    My permit expires before my appointment
  </button>

</div>

<div id="out"></div>

<script>

const out = document.getElementById('out');
const btn = document.getElementById('btn');
const box = document.getElementById('q');


function fill(el) {
  box.value = el.textContent.trim();
  go();
}


async function go(event) {

  if (event) {
    event.preventDefault();
  }

  const q = box.value.trim();

  if (!q) {
    return;
  }

  btn.disabled = true;

  out.innerHTML =
    '<div class="thinking">Searching the sources</div>';

  try {

    const r = await fetch(
      '/ask',
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          question: q
        })
      }
    );

    if (!r.ok) {

      const err =
        await r.json().catch(() => ({}));

      out.innerHTML =
        '<div class="error">' +
        esc(
          err.detail ||
          ('Request failed (' + r.status + ')')
        ) +
        '</div>';

      btn.disabled = false;
      return;
    }

    const d = await r.json();

    let html =
      '<div class="card">';

    html +=
      '<div class="answer' +
      (d.refused ? ' refused' : '') +
      '">' +
      format(d.answer) +
      '</div>';

    if (d.sources.length) {

      html +=
        '<div class="sources">' +
        '<span class="label">Sources</span>' +
        '<ul>' +

        d.sources.map(
          s =>
            '<li>' +
            esc(s.label) +
            '</li>'
        ).join('') +

        '</ul>' +
        '</div>';
    }

    html += '</div>';

    html +=
      '<div class="timing">' +
      d.elapsed_seconds +
      ' seconds' +
      '</div>';

    out.innerHTML = html;

  } catch (e) {

    out.innerHTML =
      '<div class="error">' +
      'Could not reach the service.' +
      '</div>';
  }

  btn.disabled = false;
}


function esc(s) {

  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}


/*
  Minimal markdown:
  - bold
  - bullet lists
  - numbered lists
  - paragraphs
*/

function format(text) {

  const blocks =
    esc(text)
      .split(/\\n\\s*\\n/);

  return blocks.map(block => {

    const lines =
      block
        .split('\\n')
        .map(l => l.trim())
        .filter(Boolean);

    const bulleted =
      lines.length &&
      lines.every(
        l =>
          /^[-*•]\\s+|^\\d+\\.\\s+/.test(l)
      );

    if (bulleted) {

      return (
        '<ul>' +

        lines.map(
          l =>
            '<li>' +
            bold(
              l.replace(
                /^[-*•]\\s+|^\\d+\\.\\s+/,
                ''
              )
            ) +
            '</li>'
        ).join('') +

        '</ul>'
      );
    }

    return (
      '<p>' +
      bold(lines.join(' ')) +
      '</p>'
    );

  }).join('');
}


function bold(s) {

  return s
    .replace(
      /\\*\\*(.+?)\\*\\*/g,
      '<b>$1</b>'
    )
    .replace(
      /__(.+?)__/g,
      '<b>$1</b>'
    );
}

</script>

</html>
"""