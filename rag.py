"""
Berlin RAG - Weighted Semantic + BM25 Retrieval

Retrieval:
  - 70% semantic/vector similarity
  - 30% BM25
  - BM25 combines original question + German query expansion
  - Maximum 2 chunks per source
  - Top-K = 12

Evaluation:
  - Runs selected evaluation questions
  - Generates answers
"""

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI


# ================================================================
# CONFIG
# ================================================================

BASE_DIR = Path(__file__).resolve().parent
FOLDER = BASE_DIR / "Documents"

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-5.6"

MAX_CHUNK = 750
MIN_CHUNK = 150

K = 12
MAX_PER_SOURCE = 2

SEMANTIC_WEIGHT = 0.70
BM25_WEIGHT = 0.30

BM25_ORIGINAL_WEIGHT = 0.60
BM25_EXPANDED_WEIGHT = 0.40

MAX_COMPLETION_TOKENS_EXPANSION = 200
MAX_COMPLETION_TOKENS_ANSWER = 800


# ================================================================
# OPENAI
# ================================================================

ENV_FILE = BASE_DIR / "API.env"

load_dotenv(ENV_FILE)

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY not found. Put it in your API.env file."
    )

client = OpenAI(api_key=api_key)


# ================================================================
# CHUNKING
# ================================================================

def split_hard(text, max_size=MAX_CHUNK, overlap=100):

    if len(text) <= max_size:
        return [text]

    pieces = []

    start = 0
    step = max_size - overlap

    while start < len(text):

        pieces.append(
            text[start:start + max_size]
        )

        start += step

        if start >= len(text):
            break

    return pieces


def chunk_legal(text, max_size=MAX_CHUNK):

    if len(text) <= max_size:
        return [text]

    lines = text.split("\n")

    header = lines[0] if lines else ""

    pieces = []
    current = ""

    parts = re.split(
        r"\n(?=\(\d+\))",
        text
    )

    for part in parts:

        if (
            current
            and len(current) + len(part) > max_size
        ):

            if pieces:
                pieces.append(
                    f"[{header}]\n{current}"
                )
            else:
                pieces.append(current)

            current = part

        else:

            current += (
                "\n" if current else ""
            ) + part

    if current:

        if pieces:
            pieces.append(
                f"[{header}]\n{current}"
            )
        else:
            pieces.append(current)

    final_pieces = []

    for piece in pieces:

        if len(piece) <= max_size:

            final_pieces.append(piece)

        else:

            final_pieces.extend(
                split_hard(
                    piece,
                    max_size=max_size
                )
            )

    return final_pieces


def chunk_berlin(text, max_size=MAX_CHUNK):

    headings = [
        "Verfahrensablauf",
        "Voraussetzungen",
        "Erforderliche Unterlagen",
        "Formulare",
        "Gebühren",
        "Rechtsgrundlagen",
        "Durchschnittliche Bearbeitungszeit",
        "Weiterführende Informationen",
    ]

    pattern = "|".join(
        re.escape(h)
        for h in headings
    )

    parts = [
        p.strip()
        for p in re.split(
            rf"\n(?={pattern})",
            text
        )
        if p.strip()
    ]

    if not parts:
        parts = [text]

    output = []

    for part in parts:

        if len(part) <= max_size:

            output.append(part)

        else:

            output.extend(
                split_hard(
                    part,
                    max_size=max_size
                )
            )

    return output


def merge_small(chunks, min_size=MIN_CHUNK):

    output = []

    for chunk in chunks:

        if (
            output
            and len(chunk["text"]) < min_size
            and output[-1]["source"] == chunk["source"]
	    and len(output[-1]["text"]) + 1 + len(chunk["text"]) <= MAX_CHUNK
        ):

            output[-1]["text"] += (
                "\n" + chunk["text"]
            )

        else:

            output.append(
                dict(chunk)
            )

    counts = {}

    for chunk in output:

        index = counts.get(
            chunk["source"],
            0
        )

        chunk["index"] = index

        counts[chunk["source"]] = index + 1

    return output


# ================================================================
# BUILD CORPUS
# ================================================================

def build_corpus():

    documents = []

    directories = [
        "bmg_sections",
        "aufenthg_sections",
        "sgb_sections",
        "aufenthv_sections",
        "vab_sections",
    ]

    for directory_name in directories:

        directory = FOLDER / directory_name

        if not directory.exists():

            print(
                f"WARNING: directory not found: {directory}"
            )

            continue

        for path in sorted(
            directory.glob("*.txt")
        ):

            documents.append(
                {
                    "name": path.name,
                    "text": path.read_text(
                        encoding="utf-8"
                    ),
                }
            )

    for path in sorted(
        FOLDER.glob("*.md")
    ):

        documents.append(
            {
                "name": path.name,
                "text": path.read_text(
                    encoding="utf-8"
                ),
            }
        )

    chunks = []

    for document in documents:

        name = document["name"]
        text = document["text"]

        if name.endswith(".md"):

            parts = chunk_berlin(
                text
            )

        else:

            parts = chunk_legal(
                text
            )

        for index, part in enumerate(parts):

            chunks.append(
                {
                    "text": (
                        f"[{name}]\n"
                        f"{part}"
                    ),
                    "source": name,
                    "index": index,
                }
            )

    chunks = merge_small(
        chunks
    )

    return documents, chunks


# ================================================================
# BM25
# ================================================================

def tokenize(text):

    return re.findall(
        r"\w+",
        text.lower()
    )


class BM25:

    def __init__(
        self,
        chunks,
        k1=1.5,
        b=0.75
    ):

        self.k1 = k1
        self.b = b

        self.tokens = [
            tokenize(chunk["text"])
            for chunk in chunks
        ]

        self.counts = []

        document_frequency = {}

        for document in self.tokens:

            counts = {}

            for token in document:

                counts[token] = (
                    counts.get(token, 0) + 1
                )

            self.counts.append(
                counts
            )

            for token in counts:

                document_frequency[token] = (
                    document_frequency.get(token, 0) + 1
                )

        n = len(self.tokens)

        self.idf = {
            token: np.log(
                1
                + (
                    n - frequency + 0.5
                )
                / (
                    frequency + 0.5
                )
            )
            for token, frequency
            in document_frequency.items()
        }

        lengths = [
            len(document)
            for document in self.tokens
        ]

        self.avg_len = (
            sum(lengths) / n
            if n
            else 1
        )

    def score(self, question):

        query_tokens = tokenize(
            question
        )

        scores = np.zeros(
            len(self.tokens)
        )

        for i, counts in enumerate(
            self.counts
        ):

            if not counts:
                continue

            document_length = len(
                self.tokens[i]
            )

            score = 0.0

            for token in query_tokens:

                frequency = counts.get(
                    token
                )

                if not frequency:
                    continue

                idf = self.idf.get(
                    token,
                    0
                )

                denominator = (
                    frequency
                    + self.k1
                    * (
                        1
                        - self.b
                        + self.b
                        * document_length
                        / self.avg_len
                    )
                )

                score += (
                    idf
                    * frequency
                    * (self.k1 + 1)
                    / denominator
                )

            scores[i] = score

        return scores


# ================================================================
# EMBEDDINGS
# ================================================================

def embed_batch(
    texts,
    size=100
):

    output = []

    for start in range(
        0,
        len(texts),
        size
    ):

        response = client.embeddings.create(
            model=EMBED_MODEL,
            input=texts[start:start + size]
        )

        output.extend(
            item.embedding
            for item in response.data
        )

    return np.array(
        output
    )


def embed_query(text):

    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=[text]
    )

    return np.array(
        response.data[0].embedding
    )


def cached_vectors(chunks):

    path = (
        FOLDER
        / f"vectors_{len(chunks)}.npy"
    )

    if path.exists():

        vectors = np.load(
            path
        )

        if len(vectors) == len(chunks):

            print(
                f"Using cached vectors: {path}"
            )

            return vectors

        print(
            "Cached vector count does not match "
            "current corpus. Rebuilding."
        )

    print(
        f"Creating embeddings for "
        f"{len(chunks)} chunks..."
    )

    vectors = embed_batch(
        [
            chunk["text"]
            for chunk in chunks
        ]
    )

    np.save(
        path,
        vectors
    )

    return vectors


# ================================================================
# PROMPT
# ================================================================

PROMPT = """
YOU MUST WRITE YOUR ANSWER IN Arabic.

You are an expert administrative and regulatory intelligence assistant.

Answer the user's question accurately and clearly based ONLY on the
provided SOURCES.

IMPORTANT:

1. SOURCE-BASED ANSWERING

Use only information supported by the provided SOURCES.

Do not introduce outside legal knowledge.

Do not invent missing facts.

Do not guess.


2. METADATA AND STRUCTURED DATA

Treat all information in the context as factual evidence, including:

- document metadata
- headings
- key-value fields
- lists
- tables
- costs
- fees
- conditions
- legal provisions


3. SEMANTIC BRIDGING

Users may use everyday language rather than legal terminology.

Map concepts when the underlying administrative/legal category is clearly
present in the sources.

Examples:

hotel / hostel
-> accommodation / lodging establishment

paperwork
-> required documents

lost proof
-> certificate / confirmation

move / change address
-> Anmeldung / Ummeldung


4. LEGAL MODALITY

Preserve the distinction between:

- must / shall
- may
- can
- cannot
- prohibited
- conditional requirements

Do not turn a discretionary rule into a mandatory rule.

Do not turn a conditional rule into an unconditional rule.


5. REFUSAL

Output EXACTLY:

The sources I have do not cover this.

ONLY when the core policy, legal rule, requirement, or numerical figure
needed to answer the question is absent from the provided SOURCES.

Do not refuse simply because the user's wording differs from the source.


6. WHOSE SITUATION IS THIS?

Before using a rule, identify who or what the rule is written about, and
check that the person asking falls inside that group.

Ask yourself, for each rule you are about to apply:

- Who does this rule govern? A rule about people already registered in
  Germany does not govern someone arriving from abroad. A rule about
  online registration does not govern an in-person appointment. A rule
  about family members insured through a relative does not govern a
  student insured in their own right.

- What question does this text answer? A source explaining WHO may issue
  a document does not answer WHETHER a particular person qualifies. A
  source explaining WHAT a permit is for does not answer WHETHER a change
  of course requires a new one.

If the source is about the right topic but does not address the situation
in the question, say so:

"The sources state [the rule]. They do not address [the specific
situation you asked about]."

That is a complete and useful answer. Do not close the gap yourself.


7. CITATIONS

Every factual claim must cite the exact source filename.

Example:

International students may work up to 140 full working days per year
[aufenthg_en_16b.txt].


8. ANSWER STYLE

Give a direct answer first.

Then provide a short explanation if needed.

Do not add unsupported advice.

SOURCES:

{context}

QUESTION:

{question}

Answer in Arabic.
"""


# ================================================================
# SYSTEM
# ================================================================

class System:

    def __init__(
        self,
        verbose=True
    ):

        self.documents, self.chunks = (
            build_corpus()
        )

        self.vectors = cached_vectors(
            self.chunks
        )

        self.bm25 = BM25(
            self.chunks
        )

        if verbose:

            sizes = [
                len(chunk["text"])
                for chunk in self.chunks
            ]

            print(
                f"{len(self.documents)} documents "
                f"-> {len(self.chunks)} chunks"
            )

            print(
                f"embedding index: "
                f"{self.vectors.shape}"
            )

            print(
                f"chunk size: "
                f"min={min(sizes)}, "
                f"median={int(np.median(sizes))}, "
                f"max={max(sizes)}"
            )

            print()

            print(
                "RETRIEVAL SETTINGS"
            )

            print(
                f"semantic weight: "
                f"{SEMANTIC_WEIGHT:.0%}"
            )

            print(
                f"BM25 weight: "
                f"{BM25_WEIGHT:.0%}"
            )

            print(
                f"BM25 original weight: "
                f"{BM25_ORIGINAL_WEIGHT:.0%}"
            )

            print(
                f"BM25 expanded weight: "
                f"{BM25_EXPANDED_WEIGHT:.0%}"
            )

            print(
                f"K: {K}"
            )

            print(
                f"MAX_PER_SOURCE: "
                f"{MAX_PER_SOURCE}"
            )

            print()

    # ============================================================
    # QUERY EXPANSION
    # ============================================================

    def expand_query_de(
        self,
        question
    ):

        try:

            response = client.chat.completions.create(
                model=CHAT_MODEL,
                max_completion_tokens=MAX_COMPLETION_TOKENS_EXPANSION,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Map the following user question to "
                            "German administrative and legal "
                            "search keywords.\n\n"

                            "Include relevant German legal terms, "
                            "statutory sections, and administrative "
                            "terminology when appropriate.\n\n"

                            "Examples:\n"

                            "- cost/fee -> "
                            "Gebühren kostenfrei\n"

                            "- lost paper/proof -> "
                            "Meldebescheinigung Meldebestätigung\n"

                            "- fine/penalty -> "
                            "Ordnungswidrigkeit Bußgeld § 54\n"

                            "- hotel/hostel -> "
                            "Beherbergungsstätte § 29\n"

                            "- move/change address -> "
                            "Anmeldung Ummeldung § 17 § 21\n\n"

                            f"Question: {question}\n\n"

                            "Output ONLY space-separated "
                            "German keywords."
                        )
                    }
                ]
            )

            de_terms = (
                response
                .choices[0]
                .message
                .content
                .strip()
            )

            return de_terms

        except Exception as exc:

            print(
                f"WARNING: query expansion failed: {exc}"
            )

            return ""

    # ============================================================
    # NORMALIZATION
    # ============================================================

    @staticmethod
    def minmax_normalize(
        scores
    ):

        scores = np.asarray(
            scores,
            dtype=float
        )

        minimum = scores.min()
        maximum = scores.max()

        if maximum - minimum < 1e-12:

            return np.zeros_like(
                scores
            )

        return (
            scores - minimum
        ) / (
            maximum - minimum
        )

    # ============================================================
    # SEARCH
    # ============================================================

    def search(
        self,
        question,
        k=K,
        max_per_source=MAX_PER_SOURCE
    ):

        # --------------------------------------------------------
        # Semantic
        # --------------------------------------------------------

        query_vector = embed_query(
            question
        )

        vector_norms = np.linalg.norm(
            self.vectors,
            axis=1
        )

        query_norm = np.linalg.norm(
            query_vector
        )

        semantic_scores = (
            self.vectors @ query_vector
            / (
                vector_norms
                * query_norm
                + 1e-12
            )
        )

        # --------------------------------------------------------
        # BM25 original
        # --------------------------------------------------------

        bm25_original = self.bm25.score(
            question
        )

        # --------------------------------------------------------
        # BM25 German expansion
        # --------------------------------------------------------

        de_terms = self.expand_query_de(
            question
        )

        if de_terms:

            bm25_expanded = self.bm25.score(
                de_terms
            )

        else:

            bm25_expanded = np.zeros(
                len(self.chunks)
            )

        # --------------------------------------------------------
        # Normalize BM25
        # --------------------------------------------------------

        bm25_original_norm = (
            self.minmax_normalize(
                bm25_original
            )
        )

        bm25_expanded_norm = (
            self.minmax_normalize(
                bm25_expanded
            )
        )

        # --------------------------------------------------------
        # Combine BM25
        # --------------------------------------------------------

        bm25_combined = (
            BM25_ORIGINAL_WEIGHT
            * bm25_original_norm
            +
            BM25_EXPANDED_WEIGHT
            * bm25_expanded_norm
        )

        # --------------------------------------------------------
        # Normalize semantic
        # --------------------------------------------------------

        semantic_norm = (
            self.minmax_normalize(
                semantic_scores
            )
        )

        # --------------------------------------------------------
        # Final score
        # --------------------------------------------------------

        final_scores = (
            SEMANTIC_WEIGHT
            * semantic_norm
            +
            BM25_WEIGHT
            * bm25_combined
        )

        # --------------------------------------------------------
        # Ranking
        # --------------------------------------------------------

        ranked_indices = np.argsort(
            final_scores
        )[::-1]

        # --------------------------------------------------------
        # Maximum chunks per source
        # --------------------------------------------------------

        results = []
        source_counts = {}

        for index in ranked_indices:

            source = self.chunks[index]["source"]

            count = source_counts.get(
                source,
                0
            )

            if count >= max_per_source:
                continue

            source_counts[source] = count + 1

            results.append(
                (
                    self.chunks[index],
                    float(final_scores[index]),
                    float(semantic_scores[index]),
                    float(bm25_combined[index]),
                )
            )

            if len(results) >= k:
                break

        return results

    # ============================================================
    # ASK
    # ============================================================

    def ask(
        self,
        question,
        k=K
    ):

        hits = self.search(
            question,
            k=k
        )

        context_parts = []

        for index, (
            chunk,
            final_score,
            semantic_score,
            bm25_score
        ) in enumerate(hits):

            context_parts.append(
                f"""
--- CHUNK {index + 1} [{chunk['source']}] ---
{chunk['text']}
"""
            )

        context = "\n".join(
            context_parts
        )

        response = client.chat.completions.create(
            model=CHAT_MODEL,
            max_completion_tokens=MAX_COMPLETION_TOKENS_ANSWER,
            messages=[
                {
                    "role": "user",
                    "content": PROMPT.format(
                        context=context,
                        question=question
                    )
                }
            ]
        )

        answer = (
            response
            .choices[0]
            .message
            .content
        )

        return answer, hits


# ================================================================
# REVIEW
# ================================================================

def run_review(
    system,
    questions,
    ids=None
):

    if isinstance(
        questions,
        dict
    ):

        questions = questions.get(
            "questions",
            []
        )

    if ids:

        questions = [
            question
            for question in questions
            if isinstance(question, dict)
            and question.get("id") in ids
        ]

    results = []

    for number, question in enumerate(
        questions,
        1
    ):

        if isinstance(
            question,
            dict
        ):

            question_text = question.get(
                "question",
                ""
            )

            question_id = question.get(
                "id",
                f"Q{number}"
            )

            correct_sources = question.get(
                "source_files",
                []
            )

            coverage = question.get(
                "coverage",
                "-"
            )

            expected = question.get(
                "expected"
            )

        else:

            question_text = str(
                question
            )

            question_id = f"Q{number}"

            correct_sources = []

            coverage = "-"

            expected = None

        answer, hits = system.ask(
            question_text
        )

        refused = (
            "do not cover"
            in answer.lower()
        )

        retrieved_sources = [
            chunk["source"]
            for chunk, _, _, _
            in hits
        ]

        correct_source_found = any(
            source in correct_sources
            for source in retrieved_sources
        )

        print(
            "=" * 80
        )

        print(
            f"[{number}/{len(questions)}] "
            f"{question_id}   "
            f"coverage: {coverage}"
        )

        print(
            f"Q: {question_text}"
        )

        print(
            "-" * 80
        )

        print(
            answer.strip()
        )

        print(
            "-" * 80
        )

        if expected:

            print(
                f"EXPECTED: {expected}"
            )

        print(
            "RETRIEVED TOP-12:"
        )

        for rank, (
            chunk,
            final_score,
            semantic_score,
            bm25_score
        ) in enumerate(
            hits,
            1
        ):

            print(
                f"  {rank:2d}. "
                f"final={final_score:.4f} "
                f"semantic={semantic_score:.4f} "
                f"bm25={bm25_score:.4f} "
                f"| {chunk['source']} "
                f"(chunk {chunk['index']})"
            )

        if correct_sources:

            print(
                f"CORRECT SOURCE FOUND: "
                f"{'yes' if correct_source_found else 'NO'} "
                f"(wanted: "
                f"{', '.join(correct_sources)})"
            )

        if refused:

            print(
                "FLAGS: REFUSED"
            )

        print()

        results.append(
            {
                "id": question_id,
                "coverage": coverage,
                "refused": refused,
                "retrieved_correct": correct_source_found,
                "answer": answer.strip(),
            }
        )

    # ============================================================
    # SUMMARY
    # ============================================================

    should_answer = [
        result
        for result in results
        if result["coverage"] != "none"
    ]

    should_refuse = [
        result
        for result in results
        if result["coverage"] == "none"
    ]

    print(
        "=" * 80
    )

    print(
        "SUMMARY EVALUATION"
    )

    print(
        "=" * 80
    )

    print(
        "Retrieval:"
    )

    if should_answer:

        correct_retrieval = sum(
            result["retrieved_correct"]
            for result in should_answer
        )

        print(
            f"  right source retrieved: "
            f"{correct_retrieval}"
            f"/{len(should_answer)} "
            f"("
            f"{100 * correct_retrieval / len(should_answer):.1f}%"
            f")"
        )

        answered = sum(
            not result["refused"]
            for result in should_answer
        )

        print(
            f"  produced an answer: "
            f"{answered}"
            f"/{len(should_answer)} "
            f"("
            f"{100 * answered / len(should_answer):.1f}%"
            f")"
        )

    if should_refuse:

        correct_refusals = sum(
            result["refused"]
            for result in should_refuse
        )

        print(
            f"  correctly refused: "
            f"{correct_refusals}"
            f"/{len(should_refuse)} "
            f"("
            f"{100 * correct_refusals / len(should_refuse):.1f}%"
            f")"
        )

    print()

    print(
        "NOTE:"
    )

    print(
        "This evaluation measures retrieval and refusal "
        "behavior. It does NOT automatically determine "
        "whether an answer is legally correct."
    )

    return results


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":

    args = sys.argv[1:]

    system = System()

    # ============================================================
    # ASK MODE
    # ============================================================

    if args and args[0] == "ask":

        question = " ".join(
            args[1:]
        )

        if not question:

            print(
                "Usage:"
            )

            print(
                'python rag.py ask "Your question here"'
            )

            sys.exit(1)

        answer, hits = system.ask(
            question
        )

        print()

        print(
            "ANSWER"
        )

        print(
            "=" * 80
        )

        print(
            answer
        )

        print()

        print(
            "RETRIEVED CHUNKS"
        )

        print(
            "=" * 80
        )

        for rank, (
            chunk,
            final_score,
            semantic_score,
            bm25_score
        ) in enumerate(
            hits,
            1
        ):

            print(
                f"{rank:2d}. "
                f"final={final_score:.4f} "
                f"semantic={semantic_score:.4f} "
                f"bm25={bm25_score:.4f} "
                f"| {chunk['source']} "
                f"(chunk {chunk['index']})"
            )

    # ============================================================
    # REVIEW MODE
    # ============================================================

    else:

        json_path = (
            FOLDER
            / "student_faq_80.json"
        )

        if not json_path.exists():

            raise FileNotFoundError(
                f"Could not find: {json_path}"
            )

        questions = json.loads(
            json_path.read_text(
                encoding="utf-8"
            )
        )

        wrong_ids = [
            "q23",
            "q47",
            "q53",
            "q63",
        ]

        print()

        print(
            "=" * 80
        )

        print(
            "RUNNING ONLY THE 4 WRONG QUESTIONS"
        )

        print(
            "=" * 80
        )

        print()

        print(
            "Questions:",
            ", ".join(wrong_ids)
        )

        print()

        run_review(
            system,
            questions,
            ids=wrong_ids
        )