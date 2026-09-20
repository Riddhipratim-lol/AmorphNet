"""SlowAPI rate-limiting middleware backed by Upstash Redis and circuit breakers.

Phase 13.2, 13.3 — Production API Layer
Enforces per-endpoint RPM limits and wraps cloud service dependencies with circuit breakers.
"""
