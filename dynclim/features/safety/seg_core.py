# seg_core.py
import hashlib
import sqlite3
import zlib
from typing import List, Tuple, Optional, Iterable

import numpy as np

try:
    from blingfire import text_to_sentences
    HAVE_BLING = True
except Exception:
    HAVE_BLING = False

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", "ignore")).hexdigest()

class SegmentationStore:
    """SQLite-backed cache for (start, end) offsets per (doc_hash, method, base_sig)."""
    def __init__(self, path: str = "seg_cache.sqlite"):
        self.db = sqlite3.connect(path)
        self.db.execute("""CREATE TABLE IF NOT EXISTS seg_offsets (
            doc_hash TEXT NOT NULL,
            method   TEXT NOT NULL,
            base_sig TEXT NOT NULL,
            offsets  BLOB NOT NULL,
            PRIMARY KEY (doc_hash, method, base_sig)
        )""")
        self.db.commit()

    def get(self, doc_hash: str, method: str, base_sig: str) -> Optional[np.ndarray]:
        cur = self.db.execute(
            "SELECT offsets FROM seg_offsets WHERE doc_hash=? AND method=? AND base_sig=?",
            (doc_hash, method, base_sig)
        )
        row = cur.fetchone()
        if not row:
            return None
        arr = np.frombuffer(zlib.decompress(row[0]), dtype=np.int32)
        return arr.reshape(-1, 2)

    def put(self, doc_hash: str, method: str, base_sig: str, offsets: np.ndarray) -> None:
        blob = zlib.compress(offsets.astype(np.int32).tobytes())
        self.db.execute(
            "INSERT OR REPLACE INTO seg_offsets (doc_hash, method, base_sig, offsets) VALUES (?, ?, ?, ?)",
            (doc_hash, method, base_sig, blob)
        )
        self.db.commit()

def char_offsets_from_sentences(text: str, sentences: Iterable[str]) -> np.ndarray:
    """
    Convert a list of sentence strings back to (start, end) char offsets by scanning.
    Robust to repeated sentences; advances the cursor.
    """
    offsets: List[Tuple[int, int]] = []
    cursor = 0
    n = len(text)
    for sent in sentences:
        if not sent:
            continue
        start = text.find(sent, cursor)
        if start == -1:
            # fallback: try from 0 (rare), then skip
            start = text.find(sent)
            if start == -1:
                continue
        end = start + len(sent)
        offsets.append((start, end))
        cursor = min(end, n)
    if not offsets:
        return np.zeros((0, 2), dtype=np.int32)
    return np.asarray(offsets, dtype=np.int32)

def slice_by_offsets(text: str, offsets: np.ndarray) -> List[str]:
    return [text[a:b].strip() for (a, b) in offsets.tolist()]

def filter_min_length(
    text: str,
    offsets: np.ndarray,
    min_len: int,
    unit: str = "chars",
    tokenizer=None
) -> np.ndarray:
    """
    Filter offsets by minimum length in characters (default) or tokens (if unit='tokens').
    If tokens, provide a fast Rust tokenizer (HuggingFace tokenizers).
    """
    if min_len <= 0 or offsets.size == 0:
        return offsets

    if unit == "chars" or tokenizer is None:
        lengths = np.asarray([b - a for (a, b) in offsets], dtype=np.int32)
        keep = lengths >= min_len
        return offsets[keep]

    # tokens
    counts = []
    for (a, b) in offsets:
        seg = text[a:b]
        enc = tokenizer.encode(seg)
        counts.append(len(enc.ids))
    counts = np.asarray(counts, dtype=np.int32)
    keep = counts >= min_len
    return offsets[keep]