"""Vocabulary coverage across both knowledge stores.

The system knows things in two places: the policy corpus (unstructured, reached
by retrieval) and the mock HR database (structured, reached by tools). A term
being absent from the corpus therefore means very little on its own -- employee
names, office cities and visa classes are absent from every policy document and
yet are perfectly answerable.

What *does* mean something is a content term absent from both. "Sabbatical",
"tuition" and "pet" are not in the policy corpus and not in the HR data, so no
combination of retrieval and tool calls can produce a grounded answer about
them. That is the condition this module detects.

Why this exists at all: cosine similarity was calibrated to separate in-corpus
from out-of-corpus questions and, on a harder negative set, does not -- "What
does the Helios pet insurance policy cover?" scores 0.771, above ten of the
sixteen genuine questions. The embedder maps "pet insurance" close to the real
benefits text because it is topically adjacent. Lexical evidence does not make
that mistake: the corpus contains no token "pet". See
evaluation/results/abstention_analysis.md for the measurements.
"""
from __future__ import annotations

import functools
import json
import re

from config import settings
from rag.ingest import bm25

# Words that carry no topical commitment. A question is not about "wants" or
# "exactly", so their absence from the corpus says nothing about answerability.
# Kept explicit rather than using a stemmer plus a stop list from a library: the
# set is small, auditable, and every entry is here because it appeared as a
# false signal in a measured run.
GENERIC = frozenset(
    """
    what when where which who whom whose how why does do did can could may might
    should shall must will would is are am was were be been being have has had
    get gets got give gives given take takes taken need needs want wants wanted
    tell explain describe list show say says said quote state states mean means
    allow allows allowed permit permits permitted switch change changes ask asks
    asked start starts starting begin begins happen happens fail fails failing
    apply applies cover covers covered entitle entitles entitled
    company employee employees employer staff person people
    policy policies rule rules document documents section sections
    my me mine our ours your yours their his her him she he they them it its
    this that these those there here now then still also please thanks thank
    right way ways thing things something anything nothing everything
    number numbers amount amounts exactly inside outside clear
    like about into onto over under between within during before after
    any all some each every both either neither other others
    good bad best worst more most less least much many few
    time times day days week weeks month months year years
    quickly soon immediately promptly urgently later sooner
    first second third last next previous same different
    total overall approximately roughly exactly precisely
    """.split()
)

_YEAR = re.compile(r"^(19|20)\d{2}$")

# A capitalised token that is not the first word of a sentence is a name: a
# person, a country, an office, a plan, an acronym. Names are exempt from the
# unknown-term test because they are entity references, not topics. "Brazil" is
# absent from every policy document and from the HR data, and yet "can I work
# from Brazil" is perfectly answerable -- the international policy applies
# wherever you go. Refusing on a name would be a false refusal.
#
# This does not weaken the gate: an out-of-corpus question also needs a topic,
# and topics are common nouns. "What is the tuition reimbursement cap for an
# MBA?" exempts MBA and is still caught by "tuition"; "What was Helios's Q3
# revenue?" exempts Q3 and is still caught by "revenue".
_PROPER = re.compile(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-zA-Z0-9'-]*)")
_SENTENCE_START = re.compile(r"(?:^|[.!?]\s+)([A-Za-z][a-zA-Z0-9'-]*)")


def proper_nouns(question: str) -> set[str]:
    """Capitalised tokens that are not sentence-initial, lowercased."""
    starts = {m.group(1).lower() for m in _SENTENCE_START.finditer(question)}
    found = {
        m.group(1).lower()
        for m in re.finditer(r"\b([A-Z][a-zA-Z0-9'’-]*)", question)
    }
    return {t for t in found if t not in starts}


def _roots(term: str) -> set[str]:
    """Candidate root forms of a term. Not a stemmer -- a small, auditable set.

    The failure this exists for is inflection: `blackouts` must match `blackout`
    and `switching` must match `switch`, or every conjugated verb in a question
    looks like an unknown topic. A real stemmer would be a dependency and would
    mangle the domain tokens this corpus is full of (`h-1b`, `401`, `pol-pto-001`),
    so the rules are kept few and explicit. Over-generating is safe: a candidate
    that is not a real word simply fails to match anything.
    """
    forms = {term}
    for suffix in ("'s", "s'", "es", "s", "ing", "ed", "ly", "ment", "tion"):
        if term.endswith(suffix) and len(term) - len(suffix) >= 3:
            stem = term[: -len(suffix)]
            forms.add(stem)
            # "running" -> "runn" -> "run"; "stopped" -> "stopp" -> "stop"
            if len(stem) >= 4 and stem[-1] == stem[-2]:
                forms.add(stem[:-1])
            # "having" -> "hav" -> "have"; "approving" -> "approv" -> "approve"
            if suffix in ("ing", "ed"):
                forms.add(stem + "e")
    return forms


def _normalise(term: str) -> str:
    """The single most likely root, for building vocabulary variants."""
    for suffix in ("'s", "s'", "es", "s"):
        if term.endswith(suffix) and len(term) - len(suffix) >= 4:
            return term[: -len(suffix)]
    return term


def _variants(term: str) -> set[str]:
    """The forms under which a term may legitimately appear in a vocabulary."""
    forms = set(_roots(term))
    # Hyphenated compounds ("two-week", "per-night") are known if any part is.
    forms.update(part for part in term.split("-") if len(part) > 2)
    return {f for f in forms if f}


def _is_generic(term: str) -> bool:
    return bool(_roots(term) & GENERIC)


@functools.lru_cache(maxsize=1)
def corpus_vocabulary() -> frozenset[str]:
    """Every token the BM25 index saw, plus normalised forms."""
    from rag.retrieve import _Index

    terms = set(_Index.get().bm25["postings"])
    return frozenset(terms | {_normalise(t) for t in terms})


@functools.lru_cache(maxsize=1)
def data_vocabulary() -> frozenset[str]:
    """Every token appearing anywhere in the mock HR datasets.

    Built from the raw JSON text rather than from named fields on purpose: the
    question is "does the system hold this word anywhere", and enumerating
    fields would silently miss one every time the schema grows.
    """
    terms: set[str] = set()
    for path in sorted(settings.mock_data_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        terms.update(bm25.tokenize(json.dumps(payload, ensure_ascii=False)))
    return frozenset(terms | {_normalise(t) for t in terms})


def known(term: str) -> bool:
    """True if either store contains the term, in any of its normalised forms."""
    vocabulary = corpus_vocabulary() | data_vocabulary()
    return any(form in vocabulary for form in _variants(term))


def unknown_terms(question: str) -> list[str]:
    """Content terms in ``question`` that neither knowledge store contains.

    A non-empty result means the question asks about something the system holds
    no record of, in either store -- which is a much stronger statement than a
    low similarity score, and the one that justifies refusing to answer.

    Known limitation: proper nouns are identified by capitalisation, so a
    question typed entirely in lower case loses that exemption and a novel place
    name can produce a false refusal. The cost is bounded -- the retrieval layer
    still returns its passages alongside the verdict, so the caller degrades to
    a hedged answer rather than a blank one.
    """
    names = proper_nouns(question)
    out: list[str] = []
    for term in bm25.tokenize(question):
        if len(term) <= 2 or _YEAR.match(term) or term.isdigit():
            continue
        if term in names or _is_generic(term):
            continue
        if not known(term) and term not in out:
            out.append(term)
    return out
