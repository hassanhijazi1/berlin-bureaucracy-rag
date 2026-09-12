"""
Tests for the Berlin bureaucracy RAG.

    pip install pytest httpx
    pytest -v

Most tests are offline: they check chunking, citation parsing and label
formatting without calling the API. The few that need OpenAI are marked
`live` and skipped by default:

    pytest -m live          run those too
"""

import re
import pytest
import rag
import api



# ================================================================
# FIXTURES
# ================================================================

@pytest.fixture(scope="session")
def system():
    """One corpus build for the whole session. Uses the cached vectors."""
    return rag.System(verbose=False)


# ================================================================
# CHUNKING
# ================================================================

def test_no_chunk_exceeds_the_embedding_limit(system):
    """A chunk over MAX_CHUNK blows the 8192-token embedding limit.

    This happened for real: a VAB section with no (1)(2)(3) markers
    stayed whole at 30,000 characters and the API rejected the batch.
    """
    oversized = [c for c in system.chunks if len(c["text"]) > rag.MAX_CHUNK + 50]
    assert not oversized, (
        f"{len(oversized)} chunks exceed MAX_CHUNK: "
        f"{[(c['source'], len(c['text'])) for c in oversized[:3]]}"
    )


def test_every_chunk_has_a_source_and_index(system):
    for c in system.chunks:
        assert c["source"], "chunk with no source"
        assert isinstance(c["index"], int)
        assert c["text"].strip(), f"empty chunk in {c['source']}"


def test_chunk_indices_are_contiguous_per_source(system):
    """merge_small renumbers indices; gaps would mean it went wrong."""
    seen = {}
    for c in system.chunks:
        seen.setdefault(c["source"], []).append(c["index"])
    for source, indices in seen.items():
        assert sorted(indices) == list(range(len(indices))), (
            f"{source} has non-contiguous chunk indices: {sorted(indices)}"
        )


def test_vectors_align_with_chunks(system):
    assert len(system.vectors) == len(system.chunks)
    assert system.vectors.shape[1] == 1536


def test_corpus_is_not_empty(system):
    assert len(system.documents) > 50
    assert len(system.chunks) > 400


def test_merge_small_folds_tiny_fragments():
    """The 'Gebühren\\nkeine' case: a 15-char chunk cannot be retrieved."""
    chunks = [
        {"text": "x" * 400, "source": "a.md", "index": 0},
        {"text": "Gebühren\nkeine", "source": "a.md", "index": 1},
        {"text": "y" * 400, "source": "a.md", "index": 2},
    ]
    merged = rag.merge_small(chunks, min_size=150)
    assert len(merged) == 2
    assert "keine" in merged[0]["text"]


def test_merge_small_does_not_cross_sources():
    chunks = [
        {"text": "x" * 400, "source": "a.md", "index": 0},
        {"text": "tiny", "source": "b.md", "index": 0},
    ]
    merged = rag.merge_small(chunks, min_size=150)
    assert len(merged) == 2
    assert merged[1]["source"] == "b.md"


def test_chunk_legal_splits_on_paragraph_markers():
    text = "§ 17 Anmeldung\n" + "\n".join(f"({i}) " + "x" * 300 for i in range(1, 5))
    pieces = rag.chunk_legal(text, max_size=750)
    assert len(pieces) > 1
    assert all(len(p) <= 750 for p in pieces)


def test_split_hard_never_exceeds_max():
    pieces = rag.split_hard("z" * 5000, max_size=750, overlap=100)
    assert all(len(p) <= 750 for p in pieces)
    assert "".join(pieces).count("z") >= 5000      # overlap means >=


# ================================================================
# CITATIONS
# ================================================================

@pytest.mark.parametrize("text,expected", [
    ("Register within 14 days [bmg_en_17.txt].", {"bmg_en_17.txt"}),
    ("Both apply [lea_studium.md; aufenthg_en_16b.txt].",
     {"lea_studium.md", "aufenthg_en_16b.txt"}),
    ("Three [a_b.md, c_d.txt, e.txt] here.", {"a_b.md", "c_d.txt", "e.txt"}),
    ("No citation at all.", set()),
])
def test_cited_files_extracts_filenames(text, expected):
    assert api.cited_files(text) == expected


def test_strip_citations_removes_markers_and_tidies_spacing():
    text = "You may work 140 days [aufenthg_en_16b.txt]. Student jobs excluded [lea_studium.md]."
    out = api.strip_citations(text)
    assert "[" not in out
    assert ".md" not in out
    assert out == "You may work 140 days. Student jobs excluded."


def test_strip_citations_leaves_plain_text_alone():
    text = "Nothing to strip here."
    assert api.strip_citations(text) == text


@pytest.mark.parametrize("filename,label", [
    ("bmg_de_17.txt", "BMG § 17"),
    ("bmg_en_23a.txt", "BMG § 23a"),
    ("aufenthg_en_16b.txt", "AufenthG § 16b"),
    ("aufenthv_de_47.txt", "AufenthV § 47"),
    ("sgb_de_199a.txt", "SGB V § 199a"),
    ("vab_a16b.txt", "VAB A 16b — LEA-Verfahrenshinweise"),
    ("lea_studium.md", "LEA — Aufenthaltserlaubnis zum Studium"),
    ("unknown_file.md", "unknown_file.md"),
])
def test_readable_labels(filename, label):
    assert api.readable(filename) == label


def test_german_and_english_sections_share_one_label():
    """aufenthg_de_16b and aufenthg_en_16b must not appear as two sources."""
    assert api.readable("aufenthg_de_16b.txt") == api.readable("aufenthg_en_16b.txt")


# ================================================================
# RETRIEVAL  (needs the OpenAI API)
# ================================================================

@pytest.mark.live
def test_search_returns_k_distinct_ish_chunks(system):
    hits = system.search("How long do I have to register my address?")
    assert len(hits) == rag.K
    counts = {}
    for c, *_ in hits:
        counts[c["source"]] = counts.get(c["source"], 0) + 1
    assert max(counts.values()) <= rag.MAX_PER_SOURCE


@pytest.mark.live
def test_search_scores_are_ordered(system):
    hits = system.search("Wie lange habe ich Zeit für die Anmeldung?")
    scores = [final for _, final, _, _ in hits]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.live
def test_registration_deadline_retrieves_paragraph_17(system):
    hits = system.search("How many days do I have to register my address?")
    sources = {c["source"] for c, *_ in hits}
    assert any("bmg" in s and "_17" in s for s in sources), sources


# ================================================================
# ANSWERS  (needs the OpenAI API)
# ================================================================

@pytest.mark.live
def test_known_answerable_question_is_answered(system):
    text, _ = system.ask("How long do I have to register my address in Berlin?")
    assert "do not cover" not in text.lower()
    assert re.search(r"\b(14|two weeks|zwei Wochen)\b", text, re.I), text


@pytest.mark.live
def test_out_of_scope_question_is_refused(system):
    """Broadcasting fee has no source in the corpus, by design."""
    text, _ = system.ask("How much is the Rundfunkbeitrag for students?")
    assert "do not cover" in text.lower(), text


@pytest.mark.live
def test_answer_cites_at_least_one_source(system):
    text, _ = system.ask("How many days can a student work per year?")
    assert api.cited_files(text), f"no citation in: {text[:200]}"


# ================================================================
# API
# ================================================================

def test_health_endpoint():
    from fastapi.testclient import TestClient
    client = TestClient(api.app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["chunks"] > 400


def test_home_serves_html():
    from fastapi.testclient import TestClient
    client = TestClient(api.app)
    r = client.get("/")
    assert r.status_code == 200
    assert "<h1>" in r.text


def test_empty_question_is_rejected():
    from fastapi.testclient import TestClient
    client = TestClient(api.app)
    r = client.post("/ask", json={"question": ""})
    assert r.status_code == 422          # pydantic min_length


def test_overlong_question_is_rejected():
    from fastapi.testclient import TestClient
    client = TestClient(api.app)
    r = client.post("/ask", json={"question": "x" * 5000})
    assert r.status_code == 422


def test_whitespace_only_question_is_rejected():
    from fastapi.testclient import TestClient
    client = TestClient(api.app)
    r = client.post("/ask", json={"question": "     "})
    assert r.status_code == 400


@pytest.mark.live
def test_ask_endpoint_returns_the_expected_shape():
    from fastapi.testclient import TestClient
    client = TestClient(api.app)
    r = client.post("/ask", json={"question": "How long do I have to register?"})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {
        "answer", "refused", "sources", "retrieved_count", "elapsed_seconds"
    }
    assert isinstance(body["refused"], bool)
    assert body["retrieved_count"] == rag.K
    assert "[" not in body["answer"]          # citations stripped


@pytest.mark.live
def test_refused_answer_lists_no_sources():
    from fastapi.testclient import TestClient
    client = TestClient(api.app)
    r = client.post("/ask", json={"question": "What is the Rundfunkbeitrag?"})
    body = r.json()
    if body["refused"]:
        assert body["sources"] == []
