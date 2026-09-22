from claims_triage_agent.retriever import BM25Retriever


def test_search_finds_the_relevant_cosmetic_exclusion_clause():
    retriever = BM25Retriever()
    results = retriever.search(
        "POL-1001", "is skin tag removal covered for cosmetic reasons", k=3
    )
    assert results, "expected at least one passage"
    assert results[0].chunk_id == "POL-1001-chunk1"
    assert "cosmetic" in results[0].text.lower()


def test_search_finds_the_relevant_mri_prior_auth_clause():
    retriever = BM25Retriever()
    results = retriever.search(
        "POL-2002", "brain MRI prior authorization emergency department", k=3
    )
    assert results
    assert results[0].chunk_id == "POL-2002-chunk1"


def test_search_returns_empty_for_irrelevant_query():
    retriever = BM25Retriever()
    results = retriever.search("POL-1001", "zzzznonexistentqueryterm", k=3)
    assert results == []


def test_search_unknown_policy_returns_empty_not_an_error():
    retriever = BM25Retriever()
    results = retriever.search("POL-DOES-NOT-EXIST", "anything", k=3)
    assert results == []


def test_results_are_sorted_by_score_descending():
    retriever = BM25Retriever()
    results = retriever.search("POL-1001", "knee arthroscopy prior authorization", k=5)
    scores = [p.score for p in results]
    assert scores == sorted(scores, reverse=True)
