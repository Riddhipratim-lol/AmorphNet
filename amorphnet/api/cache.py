"""Upstash Redis query caching and asynchronous ingestion task state management.

Phase 13.1 — Production API Layer
Implements 1-hour TTL answer caching keyed by SHA256(question + filters) and background job tracking.
"""
