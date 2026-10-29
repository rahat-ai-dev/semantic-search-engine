"""Download and clean the real-world corpus.

Corpus: the 20 Newsgroups collection. These are real Usenet posts written by real
people in the 1990s, grouped into 20 topics (religion, space, hockey, ...). It is
a standard public benchmark, so we also get a ground-truth label (the newsgroup)
that we use later to measure search quality honestly.

Source mirror: the `gensim-data` project on GitHub (release asset 20-newsgroups).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import random
import re
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

NEWSGROUPS_URL = (
    "https://github.com/RaRe-Technologies/gensim-data/releases/download/"
    "20-newsgroups/20-newsgroups.gz"
)
GLOVE_URL = (
    "https://github.com/RaRe-Technologies/gensim-data/releases/download/"
    "glove-wiki-gigaword-100/glove-wiki-gigaword-100.gz"
)

RAW_NEWSGROUPS = DATA_DIR / "20-newsgroups.gz"
RAW_GLOVE = DATA_DIR / "glove-wiki-gigaword-100.gz"
CORPUS_PATH = DATA_DIR / "corpus.jsonl"
QUERIES_PATH = DATA_DIR / "eval_queries.jsonl"


def download(url: str, dest: Path) -> Path:
    """Download `url` to `dest` unless it already exists."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[skip] {dest.name} already downloaded")
        return dest
    print(f"[download] {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"[done] {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
_EMAIL = re.compile(r"\S+@\S+")
_ATTRIBUTION = re.compile(r"(writes|wrote|said)\s*:?\s*$", re.IGNORECASE)


def clean_post(raw: str) -> dict | None:
    """Turn one raw Usenet post into {title, text}. Returns None if unusable.

    Steps: split header/body, keep the Subject as the title, drop quoted replies
    (lines starting with '>'), drop signatures, e-mail addresses and extra spaces.
    """
    header, _, body = raw.partition("\n\n")
    title = ""
    for line in header.splitlines():
        if line.lower().startswith("subject:"):
            title = line.split(":", 1)[1].strip()
    title = re.sub(r"^(re|fwd?)\s*:\s*", "", title, flags=re.IGNORECASE).strip()

    body = body.split("\n-- \n")[0]  # strip signature block
    kept = []
    for line in body.splitlines():
        s = line.strip()
        if not s or s.startswith(">") or s.startswith("|"):
            continue
        if _ATTRIBUTION.search(s):  # "John writes:" lines
            continue
        kept.append(_EMAIL.sub("", s))
    text = re.sub(r"\s+", " ", " ".join(kept)).strip()
    if not title or len(text) < 300:  # too short to be searchable
        return None
    return {"title": title, "text": text[:2000]}


def build_corpus(per_topic: int = 150, seed: int = 42) -> list[dict]:
    """Create corpus.jsonl (index documents) and eval_queries.jsonl (held-out test posts).

    * Index documents come from the official *train* split.
    * Evaluation queries come from the official *test* split, so the search engine
      never sees them while building the index.
    """
    download(NEWSGROUPS_URL, RAW_NEWSGROUPS)
    rng = random.Random(seed)
    by_split: dict[str, dict[str, list[dict]]] = {"train": {}, "test": {}}
    seen: set[str] = set()
    with gzip.open(RAW_NEWSGROUPS, "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            cleaned = clean_post(rec["data"])
            if cleaned is None:
                continue
            digest = hashlib.md5(cleaned["text"][:300].encode()).hexdigest()
            if digest in seen:  # remove cross-posted duplicates
                continue
            seen.add(digest)
            cleaned.update(id=rec["id"], topic=rec["topic"])
            by_split[rec["set"]].setdefault(rec["topic"], []).append(cleaned)

    corpus, queries = [], []
    for topic in sorted(by_split["train"]):
        docs = by_split["train"][topic]
        rng.shuffle(docs)
        corpus.extend(docs[:per_topic])
        tests = by_split["test"].get(topic, [])
        rng.shuffle(tests)
        queries.extend({"query": d["title"], "topic": topic} for d in tests[:25])

    for path, rows in ((CORPUS_PATH, corpus), (QUERIES_PATH, queries)):
        with open(path, "w", encoding="utf-8") as out:
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[corpus] {len(corpus)} documents, {len(queries)} evaluation queries")
    return corpus


def load_corpus() -> list[dict]:
    if not CORPUS_PATH.exists():
        raise FileNotFoundError("Run `python -m semsearch.cli download` first.")
    with open(CORPUS_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_queries() -> list[dict]:
    with open(QUERIES_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f]
