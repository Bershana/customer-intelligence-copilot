"""Policy chunking, TF-IDF retrieval, and grounded answers. Part D reuses retrieve()."""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.paths import POLICIES, RAW, ROOT
import sys

sys.path.insert(0, str(ROOT))
import mock_llm  # noqa: E402
from mock_llm import chat  # noqa: E402

INDEX = None


def chunk_words(text, size=40, overlap=10):
    words = text.split()
    step = size - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than the chunk size")
    chunks = []
    start = 0
    while start < len(words):
        end = start + size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += step
    return chunks


def load_chunks(overlap=10):
    chunks = []
    for path in sorted(POLICIES.glob("*.txt")):
        for text in chunk_words(path.read_text(encoding="utf-8"), size=40, overlap=overlap):
            chunks.append({"source": path.name, "text": text})
    return chunks


class PolicyIndex:
    def __init__(self, chunks):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform([chunk["text"] for chunk in chunks])

    def retrieve(self, query, k=3):
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix).ravel()
        order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))[:k]
        return [
            {"source": self.chunks[i]["source"], "score": float(scores[i]), "text": self.chunks[i]["text"]}
            for i in order
        ]


def build_default_index():
    global INDEX
    INDEX = PolicyIndex(load_chunks(overlap=10))
    return INDEX


def retrieve(query, k=3):
    if INDEX is None:
        build_default_index()
    return INDEX.retrieve(query, k)


def _context_message(hits, question):
    lines = ["CONTEXT:"]
    for hit in hits:
        lines.append(f"[{hit['source']}]")
        lines.append(hit["text"])
    lines.append("QUESTION:")
    lines.append(question)
    return "\n".join(lines)


def answer(question, index=None):
    chosen = index or INDEX or build_default_index()
    hits = chosen.retrieve(question, k=3)
    system = (
        "POLICY ASSISTANT. Answer only from the supplied context. "
        "If the context does not contain the answer, say that you could not find the information."
    )
    response = chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": _context_message(hits, question)},
        ]
    )
    return response.choices[0].message.content, [hit["source"] for hit in hits]


def evaluate(index):
    questions = pd.read_csv(RAW / "policy_eval.csv", keep_default_na=False)
    rows = []
    for row in questions.itertuples(index=False):
        text, sources = answer(row.question, index=index)
        rows.append(
            {
                "question": row.question,
                "expected_doc": row.expected_doc,
                "expected_fact": row.expected_fact,
                "sources": sources,
                "doc_hit": row.expected_doc in sources,
                "fact_hit": row.expected_fact in text if row.expected_fact else False,
                "fallback": text.startswith("I could not find"),
                "answer": text,
            }
        )
    return pd.DataFrame(rows)


def evaluation_scores(frame):
    answerable = frame["expected_doc"].ne("NONE")
    covered = frame.loc[answerable]
    return {
        "doc_hits": int(covered["doc_hit"].sum()),
        "fact_hits": int(covered["fact_hit"].sum()),
        "answerable": int(answerable.sum()),
        "fallback_hits": int(frame.loc[~answerable, "fallback"].sum()),
        "unanswerable": int((~answerable).sum()),
    }
