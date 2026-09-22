"""Retrieval layer used by the ``search_policy_documents`` tool.

Policy documents contain exclusions, prior-authorization rules and other
conditional clauses that are never captured in the structured
``policies.json`` table (see ``tools.lookup_policy``). The agent must be
able to find the right clause in free text and cite it, not paraphrase it
from memory -- this module is what makes that possible.

``BM25Retriever`` implements Okapi BM25 from scratch, in pure Python,
rather than depending on the third-party ``rank_bm25`` package. This
keeps the retrieval layer dependency-free (stdlib only), deterministic,
and easy to audit line-by-line -- while still satisfying the same
``Retriever`` protocol, so it is a drop-in swap for ``rank_bm25``, a
vector store, or anything else later.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from claims_triage_agent.schema import RetrievedPassage

DATA_DIR = Path(__file__).parent / "data" / "policy_documents"

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# A deliberately small stopword list: policy language is dense with
# domain terms (CPT codes, plan names) that we want to keep, so we only
# strip the highest-frequency function words rather than using a large
# generic stopword list that could remove meaningful terms.
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "in", "is", "it", "its", "of", "on", "or", "that", "the", "this",
    "to", "will", "with",
}


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


@dataclass(frozen=True)
class _Chunk:
    policy_id: str
    document_id: str
    chunk_id: str
    text: str
    tokens: list[str]


class Retriever(Protocol):
    """Protocol every retrieval backend (BM25, a vector store, ...) must satisfy."""

    def search(
        self, policy_id: str, query: str, k: int = 3
    ) -> list[RetrievedPassage]:
        ...


def _split_into_chunks(policy_id: str, document_id: str, raw_text: str) -> list[_Chunk]:
    """Split a policy document into one chunk per numbered section.

    Sections are delimited by lines starting with "Section <n> - ..."; the
    heading is kept as part of the chunk text so a retrieved passage is
    self-describing (e.g. which section it came from) when cited.
    """
    section_starts = [
        m.start() for m in re.finditer(r"(?m)^Section \d+ - .+$", raw_text)
    ]
    if not section_starts:
        chunks_text = [raw_text.strip()]
    else:
        boundaries = section_starts + [len(raw_text)]
        chunks_text = [
            raw_text[boundaries[i]:boundaries[i + 1]].strip()
            for i in range(len(section_starts))
        ]

    chunks: list[_Chunk] = []
    for i, text in enumerate(chunks_text):
        if not text:
            continue
        chunk_id = f"{document_id}-chunk{i}"
        chunks.append(
            _Chunk(
                policy_id=policy_id,
                document_id=document_id,
                chunk_id=chunk_id,
                text=text,
                tokens=_tokenize(text),
            )
        )
    return chunks


class BM25Retriever:
    """Okapi BM25 retriever over the policy-document chunks of one plan.

    Indexes are built lazily per ``policy_id`` and cached, so constructing
    the retriever is cheap and repeated searches against the same policy
    do not re-tokenize the corpus.
    """

    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        k1: float = 1.5,
        b: float = 0.75,
        min_score: float = 0.05,
    ) -> None:
        self._data_dir = data_dir
        self._k1 = k1
        self._b = b
        self._min_score = min_score
        self._index_cache: dict[str, list[_Chunk]] = {}

    def _load_chunks(self, policy_id: str) -> list[_Chunk]:
        if policy_id in self._index_cache:
            return self._index_cache[policy_id]
        path = self._data_dir / f"{policy_id}.txt"
        if not path.exists():
            self._index_cache[policy_id] = []
            return []
        raw_text = path.read_text(encoding="utf-8")
        chunks = _split_into_chunks(policy_id, path.stem, raw_text)
        self._index_cache[policy_id] = chunks
        return chunks

    def search(
        self, policy_id: str, query: str, k: int = 3
    ) -> list[RetrievedPassage]:
        chunks = self._load_chunks(policy_id)
        if not chunks:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        n_docs = len(chunks)
        doc_lengths = [len(c.tokens) for c in chunks]
        avg_doc_length = sum(doc_lengths) / n_docs

        # Document frequency per query term, for the IDF term.
        doc_freq: dict[str, int] = {}
        for term in set(query_tokens):
            doc_freq[term] = sum(1 for c in chunks if term in c.tokens)

        idf: dict[str, float] = {}
        for term, df in doc_freq.items():
            idf[term] = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)

        scored: list[tuple[float, _Chunk]] = []
        for chunk, doc_len in zip(chunks, doc_lengths, strict=True):
            score = 0.0
            for term in query_tokens:
                if term not in idf:
                    continue
                term_freq = chunk.tokens.count(term)
                if term_freq == 0:
                    continue
                numerator = term_freq * (self._k1 + 1)
                denominator = term_freq + self._k1 * (
                    1 - self._b + self._b * doc_len / avg_doc_length
                )
                score += idf[term] * (numerator / denominator)
            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        top = [pair for pair in scored[:k] if pair[0] >= self._min_score]

        return [
            RetrievedPassage(
                policy_id=chunk.policy_id,
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                score=round(score, 4),
            )
            for score, chunk in top
        ]
