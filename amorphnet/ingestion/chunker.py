"""Heterogeneous element-aware chunking engine.

Phase 5.2, 5.3, 5.4 — Tier 2 Ingestion
Splits text hierarchically (512 tokens), tables by row-groups with prepended headers,
and handles chart descriptions and footnotes.
"""
