"""Qdrant Cloud 3-prefetch RRF hybrid retriever.

Phase 9.1 — Tier 2 Hybrid Retrieval
Fuses Sparse BM25 (40) + Dense Query (40) + Dense HyDE (20) with RRF (k=60) -> Top 20 chunks.
"""
