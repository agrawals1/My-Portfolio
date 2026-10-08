"""Embedding stage. `Embedder` is a Protocol so a real model (sentence-transformers, an API) drops in.
The built-in HashingEmbedder is a dependency-free, deterministic stand-in so tests/CI never download weights."""
from __future__ import annotations

import zlib
from typing import Protocol, Sequence

import numpy as np

from persona.config import Persona
from persona.normalize import STOPWORDS, normalize_title


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> np.ndarray:  # (n, d), rows L2-normalised
        ...


class HashingEmbedder:
    """Word-stem (first 5 chars) + char-trigram features hashed with crc32 (NOT builtin hash(), which is
    salted per process and would make results differ between runs)."""

    def __init__(self, dim: int = 2048) -> None:
        self.dim = dim

    def _idx(self, feature: str) -> int:
        return zlib.crc32(feature.encode()) % self.dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in normalize_title(text).split():
                if tok in STOPWORDS:
                    continue
                out[row, self._idx("w:" + tok[:5])] += 1.0
                padded = f"#{tok}#"
                for i in range(len(padded) - 2):
                    out[row, self._idx("c:" + padded[i:i + 3])] += 0.3
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1.0, norms)


class PersonaIndex:
    """Anchors = persona name + each include keyword, embedded once per tenant. A persona's score is the
    max cosine over its anchors (a short keyword anchor is not diluted by a long description)."""

    def __init__(self, personas: Sequence[Persona], embedder: Embedder) -> None:
        self.embedder = embedder
        anchors: list[str] = []
        owner: list[int] = []
        for i, p in enumerate(personas):
            for text in (p.name, *p.include_keywords):
                anchors.append(text)
                owner.append(i)
        self.personas = list(personas)
        self._owner = np.array(owner, dtype=int)
        self._matrix = embedder.embed(anchors) if anchors else np.zeros((0, 1), dtype=np.float32)

    def scores(self, texts: Sequence[str]) -> np.ndarray:
        """(n_texts, n_personas) best-anchor cosine."""
        out = np.zeros((len(texts), len(self.personas)), dtype=np.float32)
        if not texts or not len(self._owner):
            return out
        sims = self.embedder.embed(list(texts)) @ self._matrix.T
        for pi in range(len(self.personas)):
            cols = self._owner == pi
            if cols.any():
                out[:, pi] = sims[:, cols].max(axis=1)
        return out
