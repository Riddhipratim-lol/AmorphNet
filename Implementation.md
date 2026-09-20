# AmorphNet — Master Implementation Plan & Execution Blueprint

> **Purpose-Built Financial RAG Pipeline for Annual Reports & SEC Filings**  
> *Companion operational roadmap to `architecture.md` for execution within Antigravity IDE.*

---

## 1. Executive Summary & Strategy

AmorphNet is an enterprise-grade, cloud-native Financial Retrieval-Augmented Generation (RAG) system engineered specifically for complex, multi-modal SEC filings (10-K, 10-Q, 20-F) and annual reports.

To ensure empirical rigor and avoid regressions, the project follows a **Two-Tier Implementation Strategy**:
1. **Tier 1 — Minimal Viable Baseline Pipeline & Benchmark Floor**:
   Build an initial working baseline RAG pipeline (naive PyMuPDF text extraction, naive fixed-size chunking, single-vector dense retrieval in Qdrant, and unconstrained Gemini generation). Evaluate this baseline against our golden QA dataset using RAGAS to establish the empirical performance floor.
2. **Tier 2 — Full Heterogeneous Production Pipeline**:
   Build AmorphNet’s full target architecture on top of the foundation: LlamaParse v2 multimodal extraction, element-aware heterogeneous chunking, batched metadata enrichment, dual-store indexing (Qdrant Cloud Dense + BM25 Sparse & Neo4j AuraDB Knowledge Graph), query understanding with Query-Side HyDE, 3-way RRF hybrid retrieval, graph expansion, Voyage rerank-2.5 with table-boosting, citation envelope assembly, citation-enforced generation, post-generation quote verification, Upstash Redis caching, and full ablation benchmarking.

---

## 2. Global Architecture & Cloud Integration Contract

All datastores and AI services run **exclusively on managed cloud APIs** (Zero local databases, zero Docker containers):
- **Document Parsing**: LlamaParse Cloud API (`llama-cloud-services >= 0.6.94`, API v2)
- **Dense Embeddings**: Voyage AI `voyage-4-large` (1024 dim default, 32k context window, MoE architecture)
- **Sparse Vectors**: Qdrant Cloud Native BM25 Inference (`modifier: "idf"`)
- **Vector Database**: Qdrant Cloud Cluster (`qdrant-client >= 1.19.0`)
- **Graph Database**: Neo4j AuraDB (`neo4j >= 6.3.0`)
- **Re-Ranking**: Voyage AI `rerank-2.5` (32k context window, instruction-following)
- **Generation & LLM Judges**: Google GenAI `gemini-3.5-flash-lite` (`google-genai >= 2.23.0`, `langchain-google-genai`)
- **Caching & Rate Limit State**: Upstash Redis Serverless (`upstash-redis >= 1.8.0`, `redis >= 8.1.0`)
- **Pipeline Orchestration**: LangChain (`>= 1.4.0`) + LangGraph (`>= 1.2.11`)
- **API Framework**: FastAPI (`>= 0.141.1`) + Pydantic (`>= 2.13.5`) + SlowAPI (`>= 0.1.9`)
- **Evaluation**: RAGAS (`>= 0.4.3`)

---

## 3. End-to-End Dependency & Data Flow Graph

```
[PDF Filing]
     │
     ├── (Tier 1 Baseline Path) ────────────────────────────────────────────────────────┐
     │   ├── Naive PyMuPDF Text Stream                                                  │
     │   ├── Fixed 512-Token Chunker                                                    │
     │   ├── Dense-Only Vector Indexing (`financial_chunks_baseline`)                   │
     │   └── Dense Search -> Raw Prompt -> Baseline Evaluation (RAGAS Floor)            │
     │                                                                                  │
     └── (Tier 2 Full Architecture Path)                                                │
         │                                                                              │
         ▼ (Phase 4)                                                                    │
   [Pre-Processing: SHA-256 Gate, Scan Detect, Split <=200 pgs]                         │
         │                                                                              │
         ▼ (Phase 4)                                                                    │
   [LlamaParse v2 Agentic Cloud Parser]                                                 │
         │                                                                              │
         ▼ (Phase 5)                                                                    │
   [Heterogeneous Chunker: Text, Tables, Charts, Footnotes]                             │
         │                                                                              │
         ▼ (Phase 6)                                                                    │
   [Batched Metadata Enricher: 6-8 chunks/batch -> Gemini 3.5 Flash-Lite]               │
         │                                                                              │
         ├────────────────────────────────────────┬─────────────────────────────────────┤
         ▼ (Phase 7)                              ▼ (Phase 7)                           │
   [Qdrant Cloud: Dense + BM25 Sparse]      [Neo4j AuraDB: Document/Section/Chunk Graph] │
         │                                        │                                     │
         └──────────────────┬─────────────────────┘                                     │
                            │                                                           │
 [User Question] ───────────┼───────────────────────────────────────────────────────────┘
         │                  │
         ▼ (Phase 8)        │
   [Query Analysis: Classification, Filters, Keyword BM25, Query-Side HyDE]
         │
         ▼ (Phase 9)
   [Hybrid Retrieval: Qdrant 3-Prefetch RRF (Dense 40 + Sparse 40 + HyDE 20)]
         │
         ▼ (Phase 9)
   [Graph Traversal: Footnotes, Cross-References, Adjacent Context (+10 chunks)]
         │
         ▼ (Phase 10)
   [Voyage rerank-2.5: Cross-Encoder Re-ranking + Table Boost (+0.15) -> Top 6]
         │
         ▼ (Phase 11)
   [Context Assembler: Standardized [SOURCE N] Citation Envelopes]
         │
         ▼ (Phase 11)
   [Generation: Gemini 3.5 Flash-Lite with Strict Citation JSON Schema]
         │
         ▼ (Phase 11)
   [Citation Validator: Verbatim Quote & Marker Verification Gate]
         │
         ▼ (Phase 13)
   [Upstash Redis Cache (1h TTL) & FastAPI Response Delivery]
         │
         ▼ (Phase 14)
   [Evaluation Harness: Full Ablation Benchmark vs Tier 1 Baseline Floor]
```

---

## 4. Phase-by-Phase Implementation Plan

---

### Phase 0: Environment, Configuration & Core Infrastructure Setup
**Goal**: Establish reproducible project tooling, environment validation, cloud service clients, and centralized settings.

- [x] **0.1 Virtual Environment & Package Management Setup**
  - Verify Python 3.12+ environment (`.python-version`).
  - Configure `pyproject.toml` with strict dependencies:
    - `llama-cloud-services>=0.6.94`, `voyageai>=0.5.0`, `qdrant-client>=1.19.0`, `neo4j>=6.3.0`
    - `google-genai>=2.23.0`, `langchain>=1.4.0`, `langgraph>=1.2.11`, `langchain-google-genai>=2.0.0`
    - `ragas>=0.4.3`, `fastapi>=0.141.1`, `uvicorn>=0.52.4`, `pydantic>=2.13.5`, `pydantic-settings>=2.7.0`
    - `slowapi>=0.1.9`, `redis>=8.1.0`, `upstash-redis>=1.8.0`, `aiolimiter>=1.2.1`, `tenacity>=9.0.0`
    - `pymupdf>=1.25.0`, `pytest>=8.3.0`, `pytest-asyncio>=0.24.0`, `httpx>=0.28.0`
  - Install dependencies via `uv sync` or pip.
- [x] **0.2 Directory Layout Creation**
  - Build directory tree strictly conforming to Section 17 of `architecture.md`:
    - `amorphnet/ingestion/`
    - `amorphnet/retrieval/`
    - `amorphnet/generation/`
    - `amorphnet/graph/`
    - `amorphnet/evaluation/`
    - `amorphnet/api/routes/`
    - `evaluation/`
    - `scripts/`
    - `tests/unit/`, `tests/integration/`
- [x] **0.3 Configuration & Secrets Management (`amorphnet/config.py`)**
  - Implement Pydantic `Settings` class reading from `.env`.
  - Validate all cloud API keys: `LLAMA_CLOUD_API_KEY`, `VOYAGE_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `GOOGLE_API_KEY`, `UPSTASH_REDIS_REST_URL`, `UPSTASH_REDIS_REST_TOKEN`.
  - Provide runtime constants: `GEMINI_RPM_LIMIT=14`, `MAX_CONCURRENT_INGESTIONS=3`, `QUERY_TIMEOUT_SECONDS=30`, `CACHE_TTL_SECONDS=3600`, `LLAMAPARSE_MAX_CONCURRENCY=3`, `VOYAGE_MAX_RETRIES=5`, `EVAL_MODE="smoke"`.
  - Create `.env.example` with documented defaults.
- [x] **0.4 Shared Cloud Client Singletons & Rate Limiting Guardrails**
  - Implement centralized client factory modules with connection pooling and resilient backoff:
    - Voyage AI AsyncClient with `max_retries=5` and `aiolimiter.AsyncLimiter(max_rate=50, time_period=60)`.
    - Google GenAI Client with global `AsyncLimiter(max_rate=14, time_period=60)` for Free Tier protection.
    - Qdrant Cloud Client (`AsyncQdrantClient`).
    - Neo4j AuraDB Async Driver (`AsyncGraphDatabase.driver`).
    - Upstash Redis async client (`upstash-redis` / `redis.asyncio`).
- [x] **0.5 Service Connectivity Verification Script (`scripts/check_services.py`)**
  - Write test harness verifying active ping and authentication to Qdrant Cloud, Neo4j AuraDB, Voyage AI, Google GenAI, and Upstash Redis.

**Inputs**: `.env` configuration file, API credentials.  
**Outputs**: Validated connection clients, populated `config.py`, verified directory structure.  
**Interconnectivity**: All downstream modules import settings and initialized clients from `amorphnet.config` and client singletons.

---

### Phase 1: Golden Evaluation Dataset & Evaluation Harness Foundation
**Goal**: Build the ground truth evaluation dataset and RAGAS harness before pipeline construction, ensuring objective measurement from day one.

- [ ] **1.1 Curate Golden Financial QA Dataset (`evaluation/golden_dataset.json`)**
  - Curate 100 high-complexity QA pairs from 5–6 representative annual filings (e.g., Apple, Microsoft, Amazon, Alphabet, Tesla 10-Ks).
  - Enforce query distribution specified in Section 15.1:
    - 30% `factual_numeric` (multi-row financial statements, footnote reconciliation)
    - 25% `trend_comparative` (multi-year revenue/margin changes)
    - 25% `semantic_risk` (MD&A qualitative risk disclosures, regulatory factors)
    - 20% `cross_reference` (specific "Note X" disclosures, lease obligations, segment reporting)
  - Populate each record with schema: `question`, `ground_truth`, `reference_chunks`, `filing_type`, `fiscal_year`, `company_name`.
  - Formulate a 10-question stratified subset for `smoke_eval` mode (preserving the exact 30/25/25/20 distribution).
- [ ] **1.2 Build RAGAS Evaluator Harness (`amorphnet/evaluation/ragas_runner.py`)**
  - Implement sequential evaluator compatible with Ragas v0.4.3+:
    - Attach LangChain `InMemoryRateLimiter(requests_per_second=0.2)` (12 RPM) to `ChatGoogleGenAI(model="gemini-3.5-flash-lite")`.
    - Wrap judge LLM with `LangchainLLMWrapper`.
    - Wrap `VoyageAIEmbeddings(model="voyage-4-large")` with `LangchainEmbeddingsWrapper`.
    - Initialize core metric instances: `LLMContextPrecisionWithReference`, `LLMContextRecall`, `Faithfulness`, `ResponseRelevancy`, `AnswerCorrectness`.
    - Configure `RunConfig(max_workers=1, timeout=60, max_retries=5, max_wait=60)`.
- [ ] **1.3 Build Evaluation Execution Script (`scripts/run_evaluation.py`)**
  - Implement CLI supporting `--mode smoke` (10 items, ~50 LLM calls) and `--mode full` (100 items, ~500 calls).
  - Format output metrics into console summary and persist results to `evaluation/ablation_results.json`.

**Inputs**: Curated 10-K/filing PDFs, ground truth answers.  
**Outputs**: `evaluation/golden_dataset.json`, functional `ragas_runner.py`, verified evaluation CLI.  
**Interconnectivity**: Provides the standard benchmark harness used to evaluate Tier 1 Baseline (Phase 3) and Tier 2 Full Pipeline (Phase 14).

---

### Phase 2: Tier 1 Baseline Pipeline Implementation
**Goal**: Build a functional, minimal-viable RAG baseline using naive PDF text extraction, fixed-size chunking, single-vector dense Qdrant search, and unconstrained LLM generation.

- [ ] **2.1 Baseline Document Ingestion & Chunking (`amorphnet/ingestion/baseline_ingest.py`)**
  - Implement naive text extraction using PyMuPDF (`fitz` character stream).
  - Chunk extracted text into uniform 512-token chunks with 50-token overlap using Voyage tokenizer.
  - Retain minimal metadata: `document_id`, `page_number`, `chunk_id`, `text`.
- [ ] **2.2 Baseline Vector Indexing (`amorphnet/ingestion/baseline_indexer.py`)**
  - Create dedicated Qdrant collection: `financial_chunks_baseline`.
  - Embed chunks in batches of 128 using Voyage AI `voyage-4-large` (`input_type="document"`).
  - Upsert dense vectors (1024-dim, Cosine) into Qdrant Cloud without sparse vectors or metadata enrichment.
- [ ] **2.3 Baseline Dense Retrieval (`amorphnet/retrieval/baseline_retriever.py`)**
  - Embed incoming raw user query with `voyage-4-large` (`input_type="query"`).
  - Query `financial_chunks_baseline` using single dense vector search for top-6 chunks (no BM25, no HyDE, no re-ranking, no graph expansion).
- [ ] **2.4 Baseline Generation (`amorphnet/generation/baseline_generator.py`)**
  - Assemble retrieved chunk texts into a naive concatenated prompt block.
  - Send context and question to Gemini 3.5 Flash-Lite without citation envelopes or JSON schema validation.
  - Return plain-text string answer.

**Inputs**: Sample 10-K PDFs, user queries.  
**Outputs**: Populated `financial_chunks_baseline` collection, end-to-end baseline query function.  
**Interconnectivity**: Establishes the unaugmented reference implementation to compare against AmorphNet’s full pipeline.

---

### Phase 3: Tier 1 Baseline Evaluation & Benchmark Floor Establishment
**Goal**: Run baseline evaluation, capture baseline metrics, and document the empirical performance gap.

- [ ] **3.1 Ingest Benchmark Filings into Baseline Index**
  - Run `baseline_ingest.py` on the 5–6 representative annual filings used for the golden dataset.
- [ ] **3.2 Execute Baseline Evaluation Run**
  - Run `scripts/run_evaluation.py --pipeline baseline --mode smoke` (and `--mode full` if quota permits).
  - Generate baseline answer responses and retrieved contexts for all golden dataset questions.
  - Compute RAGAS scores: `ContextPrecision`, `ContextRecall`, `Faithfulness`, `AnswerRelevance`, `AnswerCorrectness`.
- [ ] **3.3 Document Baseline Performance Floor**
  - Record baseline scores in `evaluation/ablation_results.json` under key `"baseline"`.
  - Identify key failure modes (table truncation, missing footnotes, hallucinated numbers, lack of citations).

**Inputs**: Populated baseline index, `golden_dataset.json`.  
**Outputs**: Baseline evaluation scores in `evaluation/ablation_results.json`, failure mode baseline report.  
**Interconnectivity**: Sets the baseline target that Tier 2 must surpass across all five RAGAS metrics.

---

### Phase 4: Phase 0 & Phase 1 Ingestion — Structure-Aware Parsing & Document Registry
**Goal**: Implement robust PDF pre-processing, deduplication gates, concurrency throttling, and LlamaParse v2 multimodal parsing.

- [ ] **4.1 Document Registry (`amorphnet/ingestion/document_registry.py`)**
  - Implement SQLite/PostgreSQL metadata registry with schema:
    - `document_id` (UUID primary key), `filename`, `sha256_hash`, `company_name`, `fiscal_year`, `filing_type`, `page_count`, `ingestion_date`, `status` ("pending", "parsing", "chunking", "enriching", "indexed", "failed").
  - Implement deduplication gate: compute SHA-256 of raw PDF bytes; abort ingestion if duplicate hash exists.
- [ ] **4.2 PDF Pre-Processing & Rate-Limit Guardrails (`amorphnet/ingestion/pdf_preprocessor.py`)**
  - Validate password protection status; throw explicit error if encrypted.
  - Detect scanned image-only PDFs via PyMuPDF `page.get_images()` and set `scan_mode=True`.
  - Extract basic document metadata (title, author, creation date) from XMP metadata.
  - Enforce page cap: check page count; if > 500 pages, split into sequential batch PDFs of <= 200 pages.
- [ ] **4.3 LlamaParse v2 Cloud Client Integration (`amorphnet/ingestion/llamaparse_client.py`)**
  - Integrate `llama-cloud-services` SDK with API v2.
  - Configure parser parameters:
    - `parse_mode="parse"`, `output_format="markdown"`, `language="en"`, `premium_mode=True`
    - `skip_diagonal_text=True`, `take_screenshot=False`, `invalidate_cache=False`
    - `page_separator="---PAGE {page_num}---"`
  - Inject domain-specific financial system prompt instructions (preserving tables as markdown, table captions, chart descriptions, footnote anchors, column reading order).
  - Wrap job submission and polling in `asyncio.Semaphore(3)` (respecting 5-job concurrency ceiling).
  - Implement `tenacity` exponential backoff (2s initial, 30s max) on HTTP 429.
  - Output standardized `LlamaParseResult` per page: `page_number`, `markdown_content`, `element_type_map` ("text", "table", "chart_description", "footnote"), `raw_images`.

**Inputs**: Raw PDF binary stream, filing metadata (`company_name`, `fiscal_year`, `filing_type`).  
**Outputs**: Registered document record, page-by-page structured markdown with element maps.  
**Interconnectivity**: Passes parsed page structures to Phase 5 Heterogeneous Chunker.

---

### Phase 5: Phase 2 Chunking Strategy — Heterogeneous Element-Aware Chunker
**Goal**: Implement element-specific chunking for prose, financial tables, charts, and footnotes with standardized metadata schemas.

- [ ] **5.1 Standardized Chunk Data Model (`amorphnet/ingestion/chunk_schema.py`)**
  - Define `Chunk` Pydantic model with fields:
    - `chunk_id` (UUID), `document_id`, `company_name`, `ticker`, `filing_type`, `fiscal_year`, `fiscal_quarter`
    - `page_number`, `section_header`, `element_type` (`text` | `table` | `chart` | `footnote`)
    - `table_caption`, `parent_anchor`, `chunk_index`, `token_count`, `char_offset_start`, `char_offset_end`, `text`
- [ ] **5.2 Hierarchical Text Splitter (`amorphnet/ingestion/chunker.py`)**
  - Narrative prose splitter:
    - Primary split on markdown section headers (`##`, `###`).
    - Secondary split at sentence boundaries targeting 512 tokens using Voyage AI tokenizer (`voyageai.get_tokenizer()`).
    - Enforce 50-token sliding window overlap between adjacent chunks in the same section.
    - Hard ceiling cap at 768 tokens.
- [ ] **5.3 Table Chunking Engine (`amorphnet/ingestion/chunker.py`)**
  - Detect and isolate markdown tables.
  - Implement three-tier table chunking rules:
    - **Small tables** (<= 300 tokens): Single chunk with full table + table caption.
    - **Medium tables** (301–800 tokens): Split row-group-wise by semantic categories (e.g. Revenue rows, Expense rows) based on indentation/headers; prepend full column header row to every chunk.
    - **Large tables** (> 800 tokens): Split into chunks of <= 10 rows each, always prepending the full column header row.
  - Store table caption in chunk payload (`table_caption`) for index filtering.
- [ ] **5.4 Chart & Footnote Chunking (`amorphnet/ingestion/chunker.py`)**
  - **Chart chunks**: Preserve LlamaParse vision-generated structured description as a single chunk (100–300 tokens), tagged `element_type="chart"`.
  - **Footnote chunks**: Chunk individually per footnote; extract reference marker and attach `parent_anchor` pointing to the annotating text chunk.

**Inputs**: Parsed markdown pages with `element_type_map`.  
**Outputs**: List of standardized `Chunk` objects categorized by element type.  
**Interconnectivity**: Chunks are forwarded to Phase 6 for metadata enrichment before indexing.

---

### Phase 6: Phase 3 Metadata Enrichment — Batched Extraction & Breadcrumbs
**Goal**: Enrich chunks with financial entity metadata and hierarchical breadcrumbs while strictly respecting LLM rate limits via batching.

- [ ] **6.1 Batched Financial Entity Enricher (`amorphnet/ingestion/metadata_enricher.py`)**
  - Implement batching utility grouping 6–8 consecutive chunks per request.
  - Define structured output Pydantic schemas:
    - `ChunkMetadata`: `chunk_id`, `financial_metrics: list[str]`, `fiscal_periods: list[str]`, `currency_and_scale: str | None`, `is_comparative: bool`.
    - `BatchEnrichmentResponse`: `enriched_chunks: list[ChunkMetadata]`.
  - Call Gemini 3.5 Flash-Lite with `response_schema=BatchEnrichmentResponse` and system prompt instructing entity extraction and scale preservation (e.g. "USD in millions").
  - Enforce client-side rate limiting: `aiolimiter.AsyncLimiter(max_rate=14, time_period=60)` with `tenacity` exponential retry on `RESOURCE_EXHAUSTED` (429).
  - Merge extracted entity fields back into chunk metadata payload.
- [ ] **6.2 Section Hierarchy & Breadcrumb Builder**
  - Parse markdown headings to construct breadcrumb hierarchy (e.g. `"Part II > Item 8 > Note 3 — Revenue Recognition"`).
  - Assign breadcrumb path to chunk metadata `breadcrumb` and prepare section tree representation for Neo4j.

**Inputs**: Raw `Chunk` objects from Phase 5.  
**Outputs**: Enriched `Chunk` objects with financial metrics, periods, currency/scale, and section breadcrumbs.  
**Interconnectivity**: Enriched chunks are routed to Phase 7 Dual-Store Indexing (Qdrant Cloud & Neo4j AuraDB).

---

### Phase 7: Phase 4 Dual-Store Indexing — Qdrant Cloud & Neo4j AuraDB
**Goal**: Create dual-store cloud indexes: Qdrant named dense (Voyage) + sparse (BM25) vector collection, and Neo4j knowledge graph.

- [ ] **7.1 Qdrant Collection Initialization (`amorphnet/ingestion/qdrant_indexer.py`)**
  - Create collection `financial_chunks` with named vectors:
    - `dense`: `size=1024`, `distance=Cosine`, `HnswConfigDiff(m=16, ef_construct=200)`.
    - `sparse`: `modifier=models.Modifier.IDF` (Qdrant native server-side BM25 inference).
  - Create payload keyword and numeric indexes:
    - `company_name` (keyword), `ticker` (keyword), `filing_type` (keyword)
    - `fiscal_year` (integer), `fiscal_quarter` (keyword), `element_type` (keyword)
    - `page_number` (integer), `section_header` (text full-text)
- [ ] **7.2 Batched Embedding & Qdrant Ingestion**
  - Batch chunk texts into groups of <= 128 chunks for Voyage AI `voyage-4-large` (`input_type="document"`).
  - Upsert points into Qdrant Cloud in batches of 100 points, concurrency-capped via `asyncio.Semaphore(5)`.
- [ ] **7.3 Neo4j AuraDB Knowledge Graph Construction (`amorphnet/ingestion/neo4j_indexer.py`)**
  - Establish constraints and indexes in Neo4j AuraDB:
    - Unique constraints on `Document.document_id`, `Section.section_id`, `Chunk.chunk_id`, `FinancialEntity.name`.
  - Ingest graph nodes:
    - `(:Document)`: `document_id`, `company_name`, `filing_type`, `fiscal_year`
    - `(:Section)`: `section_id`, `document_id`, `header_text`, `breadcrumb`, `page_number`, `element_type`
    - `(:Chunk)`: `chunk_id`, `section_id`, `element_type`, `page_number`, `token_count`
    - `(:FinancialEntity)`: `name`, `entity_type`
  - Ingest structural and semantic relationships:
    - `(:Document)-[:HAS_SECTION]->(:Section)`
    - `(:Section)-[:HAS_CHILD_SECTION]->(:Section)`
    - `(:Section)-[:HAS_CHUNK]->(:Chunk)`
    - `(:Chunk)-[:MENTIONS]->(:FinancialEntity)`
    - `(:Section)-[:NEXT_SECTION]->(:Section)`
    - `(:Chunk)-[:FOOTNOTE_OF]->(:Chunk)` (mapped from `parent_anchor`)
- [ ] **7.4 Cross-Reference Detection Engine (`amorphnet/ingestion/neo4j_indexer.py`)**
  - Scan chunk text using regex rules: `"See Note \d+"`, `"as described in Note \d+"`, `"Refer to [A-Z][a-z]+ \d+"`, `"discussed in Item \d+"`, `"as shown in [Tt]able \d+"`.
  - Resolve matching sections/chunks and insert `(:Chunk)-[:REFERENCES]->(:Chunk)` edges into Neo4j.

**Inputs**: Enriched chunks, document metadata, section tree.  
**Outputs**: Fully populated Qdrant Cloud collection `financial_chunks` and Neo4j AuraDB graph.  
**Interconnectivity**: Forms the dual datastore queried in Phase 9 Hybrid Retrieval.

---

### Phase 8: Phase 5 Query Understanding, Routing & Query-Side HyDE
**Goal**: Classify queries, extract structured payload filters, generate BM25 keyword variants, and produce Query-Side HyDE embeddings.

- [ ] **8.1 Query Analysis Structured Model (`amorphnet/retrieval/query_classifier.py`)**
  - Define Pydantic schema `QueryAnalysis`:
    - `query_type`: Enum (`factual_numeric`, `trend_comparative`, `semantic_risk`, `cross_reference`)
    - `company_name`: str | None
    - `ticker`: str | None
    - `fiscal_year`: int | None
    - `filing_type`: str | None
    - `preferred_element_type`: str | None (`table`, `text`, `chart`)
    - `bm25_keywords`: str (keyword-dense search string)
    - `hypothetical_answer`: str (Query-Side HyDE passage simulating a financial filing excerpt)
- [ ] **8.2 Consolidated Query Understanding Node**
  - Execute a single structured Gemini 3.5 Flash-Lite call with `QueryAnalysis` schema to minimize latency and token consumption.
  - Construct Qdrant `Filter` conditions from extracted metadata (`company_name`, `fiscal_year`, `filing_type`).
- [ ] **8.3 Multi-Representation Embedding Generation (`amorphnet/retrieval/query_expander.py`)**
  - Generate dense embedding for original user query via Voyage `voyage-4-large` (`input_type="query"`).
  - Generate dense embedding for the hypothetical answer fragment (HyDE) via Voyage `voyage-4-large` (`input_type="query"`).
  - Prepare sparse BM25 query text for Qdrant server-side inference.

**Inputs**: Raw user question string, optional client filters.  
**Outputs**: `QueryAnalysis` object, dense query vector, HyDE dense vector, BM25 query string, Qdrant payload filter.  
**Interconnectivity**: Feeds representations directly into Phase 9 Hybrid Retrieval.

---

### Phase 9: Phase 6 & Phase 7 Hybrid Retrieval & Neo4j Graph Expansion
**Goal**: Execute 3-way RRF fused retrieval in Qdrant Cloud followed by structural graph traversal in Neo4j AuraDB.

- [ ] **9.1 Qdrant 3-Prefetch RRF Query (`amorphnet/retrieval/qdrant_retriever.py`)**
  - Construct Qdrant `query_points` request with three prefetches fused server-side via `RrfQuery(rrf=Rrf(k=60))`:
    - **Prefetch 1 (Sparse BM25)**: `using="sparse"`, limit 40, payload filter applied.
    - **Prefetch 2 (Dense Query)**: `using="dense"`, limit 40, payload filter applied.
    - **Prefetch 3 (Query-Side HyDE Dense)**: `using="dense"`, limit 20, payload filter applied.
  - Return top 20 fused chunks with full payload metadata.
- [ ] **9.2 Neo4j Knowledge Graph Expansion (`amorphnet/retrieval/graph_retriever.py`)**
  - For each chunk in the Qdrant top-20 pool, execute graph expansion queries:
    - Traverse `[:FOOTNOTE_OF]` to fetch associated footnote chunks (limit 3 per chunk).
    - Traverse `[:REFERENCES]` to retrieve cross-referenced section chunks (limit 2 per chunk).
    - Traverse `[:NEXT_SECTION]` to fetch preceding (-1) and succeeding (+1) chunks for sequential continuity.
  - Collect up to ~10 additional structural chunks.
- [ ] **9.3 Candidate Pool Consolidation & Baseline RRF Assignment**
  - Deduplicate combined chunks by `chunk_id`.
  - Assign graph-expansion chunks a baseline rank position of 21 in each ranking list to compute their RRF score.
  - Output candidate pool of 30 unique chunks.

**Inputs**: Query embeddings, BM25 keywords, payload filters.  
**Outputs**: Candidate pool of 30 ranked chunks.  
**Interconnectivity**: Passes candidate pool to Phase 10 Re-ranking.

---

### Phase 10: Phase 8 Re-ranking — Voyage rerank-2.5 with Table-Aware Boosting
**Goal**: Apply cross-encoder semantic re-ranking with domain-specific tabular score adjustment.

- [ ] **10.1 Voyage rerank-2.5 Integration (`amorphnet/retrieval/reranker.py`)**
  - Initialize Voyage `AsyncClient(max_retries=5, timeout=30.0)`.
  - Dispatch candidate pool (30 chunks) and original user query to model `rerank-2.5`.
  - Retrieve cross-encoder relevance scores for each candidate chunk.
- [ ] **10.2 Table-Aware Score Boosting Logic**
  - Check `query_type` from Phase 8:
    - If `query_type == "factual_numeric"`, apply a `+0.15` score boost (or `score * 1.15`) to any chunk with `element_type == "table"`.
  - Sort candidate chunks by boosted score.
  - Select top 6 highest-scoring chunks.

**Inputs**: 30 candidate chunks, original user query, `query_type`.  
**Outputs**: Top 6 re-ranked chunks.  
**Interconnectivity**: Passes top-6 chunks to Phase 11 Context Assembly and Generation.

---

### Phase 11: Phase 9 & Phase 10 Context Assembly, Citation-Enforced Generation & Post-Validation
**Goal**: Format citation envelopes, enforce strict citation attribution via Gemini 3.5 Flash-Lite, and validate quotes post-generation.

- [ ] **11.1 Standardized Citation Envelope Formatter (`amorphnet/generation/context_assembler.py`)**
  - Wrap each of the top 6 chunks in an unambiguous citation block:
    ```
    [SOURCE N]
    Company: {company_name} | Filing: {filing_type} | Fiscal Year: {fiscal_year} | Page: {page_number}
    Section: {section_header}
    Element Type: {element_type}
    
    {chunk_text}
    [END SOURCE N]
    ```
  - Concatenate envelopes into a single structured context string.
- [ ] **11.2 Citation-Enforced Generation (`amorphnet/generation/generator.py`)**
  - Define structured Pydantic response schema:
    - `Citation`: `source_number: int`, `company_name: str`, `filing_type: str`, `fiscal_year: int`, `page_number: int`, `quote: str`.
    - `AnswerSchema`: `answer: str`, `citations: list[Citation]`, `has_answer: bool`, `confidence: float`, `caveats: list[str]`.
  - Configure Gemini 3.5 Flash-Lite call:
    - System instruction enforcing mandatory `[SOURCE N]` tags, prohibiting unsupported inferences, and mandating row/column specifications for tables.
    - Parameters: `temperature=0.1`, `max_output_tokens=1024`, `response_mime_type="application/json"`, `response_schema=AnswerSchema`.
    - Rate limiter: wrapped in `gemini_limiter` (`AsyncLimiter(14, 60)`) with `tenacity` retry on `APIError`.
- [ ] **11.3 Post-Generation Citation Validation Gate (`amorphnet/generation/citation_validator.py`)**
  - Perform fast programmatic string-matching validation:
    - Extract all `[SOURCE N]` references from `answer`.
    - Validate that each `N` has a matching entry in `citations`.
    - Verify that `quote` in each citation exists verbatim in the corresponding source chunk text.
    - If validation fails, log warning and sanitize answer (or strip invalid claims).

**Inputs**: Top 6 re-ranked chunks, original question.  
**Outputs**: Validated `AnswerSchema` JSON response with verified citations.  
**Interconnectivity**: Output is cached in Redis (Phase 13) and evaluated via RAGAS (Phase 14).

---

### Phase 12: LangGraph Stateful Orchestration Pipeline
**Goal**: Wire all query processing, retrieval, re-ranking, and generation steps into a robust, observable LangGraph state machine.

- [ ] **12.1 Define Pipeline State Schema (`amorphnet/graph/state.py`)**
  - Implement `QueryState(TypedDict)`:
    - `original_query: str`, `query_type: str`, `expanded_queries: list[str]`, `filters: dict`
    - `candidate_chunks: list[Chunk]`, `reranked_chunks: list[Chunk]`, `context_string: str`
    - `answer: str`, `citations: list[Citation]`, `confidence_score: float`, `validation_passed: bool`
- [ ] **12.2 Construct LangGraph Workflow (`amorphnet/graph/langgraph_pipeline.py`)**
  - Define nodes: `analyze_query_node`, `hybrid_retrieval_node`, `graph_expansion_node`, `rerank_node`, `assemble_context_node`, `generate_answer_node`, `validate_citations_node`.
  - Define conditional edges: route based on `has_answer` and validation status.
  - Compile pipeline into an executable graph app (`amorphnet_pipeline = workflow.compile()`).
- [ ] **12.3 Unit & Integration Testing of Graph Traversal**
  - Test end-to-end execution flow with mocked and live components (`tests/integration/test_query_pipeline.py`).

**Inputs**: Initialized LangGraph state.  
**Outputs**: Compiled stateful RAG pipeline.  
**Interconnectivity**: Serves as the primary execution engine invoked by FastAPI routes and the evaluation harness.

---

### Phase 13: Phase 12 Production API Layer, Cloud Caching & Multi-Tier Throttling
**Goal**: Expose FastAPI endpoints with SlowAPI rate limiting, Upstash Redis serverless caching, background task management, and circuit breakers.

- [ ] **13.1 Upstash Redis Serverless Caching & Task State (`amorphnet/api/cache.py`)**
  - Implement query caching: cache validated `AnswerSchema` for 1 hour (`TTL=3600`) keyed by `SHA256(question + json(filters))`.
  - Implement async ingestion task tracking: store job status in Redis hash `HSET task:{task_id}`.
- [ ] **13.2 Multi-Tier Rate Limiting Middleware (`amorphnet/api/middleware.py`)**
  - Configure `slowapi` inbound rate limiting backed by Upstash Redis:
    - `POST /query`: 10 req/min per client key.
    - `POST /ingest`: 3 req/min per client key (enforce max 3 active concurrent background jobs).
    - `POST /evaluate`: 1 req/hour per key (admin protected).
  - Return HTTP 429 with standard `Retry-After` headers.
- [ ] **13.3 Circuit Breakers for Cloud Dependencies**
  - Implement lightweight circuit breaker pattern around Voyage AI and Qdrant Cloud calls.
  - Fall back to cached responses or return structured service degradation error responses without hanging requests.
- [ ] **13.4 FastAPI Application Endpoints (`amorphnet/api/main.py`)**
  - `POST /ingest`: Upload binary PDF, register document, enqueue async background parsing/indexing task.
  - `GET /ingest/{document_id}/status`: Poll status of background ingestion.
  - `POST /query`: Execute compiled LangGraph pipeline (checking cache first).
  - `POST /evaluate`: Trigger evaluation harness (`mode="smoke" | "full"`).
  - `GET /health`: Active health probe checking Qdrant Cloud, Neo4j AuraDB, and Upstash Redis.

**Inputs**: HTTP requests, uploaded files, query JSON.  
**Outputs**: JSON API responses, background task tracking.  
**Interconnectivity**: External interface for users and automated clients.

---

### Phase 14: Phase 11 Full Pipeline Evaluation, Ablation Benchmark & Final Delivery
**Goal**: Run comprehensive RAGAS evaluation across all 6 ablation configurations, prove superiority over the Tier 1 baseline, and produce the final benchmark report.

- [ ] **14.1 Ablation Study Execution Framework (`amorphnet/evaluation/ablation.py`)**
  - Implement automated runner executing the 6 architectural variants defined in Section 15.4:
    - `baseline`: Naive text, dense-only, no re-ranking, unconstrained generation (from Tier 1).
    - `+bm25`: Dense + BM25 hybrid; no re-ranking.
    - `+rerank`: Dense + BM25 + Voyage rerank-2.5; no graph.
    - `+graph`: Dense + BM25 + Voyage rerank-2.5 + Neo4j graph expansion.
    - `+hyde`: Dense + BM25 + Voyage rerank-2.5 + Neo4j + Query-Side HyDE.
    - `full`: Complete AmorphNet pipeline with table score boosting and citation validation.
- [ ] **14.2 Benchmark Run & Quota Management**
  - Run `scripts/run_evaluation.py --mode smoke` (10 questions, ~50 LLM calls) for development validation.
  - Run full benchmark on 100 questions once configured with Pay-As-You-Go API credentials.
- [ ] **14.3 Metrics Verification vs Architecture Targets**
  - Verify metrics against canonical targets from Section 15.2:
    - `ContextPrecision >= 0.75`
    - `ContextRecall >= 0.80`
    - `Faithfulness >= 0.90` (Zero tolerance for financial hallucination)
    - `AnswerRelevance >= 0.80`
    - `AnswerCorrectness >= 0.75`
- [ ] **14.4 Persist Benchmark Report & Update Documentation**
  - Write full comparative metrics to `evaluation/ablation_results.json`.
  - Generate comparative visualization/summary markdown table demonstrating quantifiable delta from baseline to full pipeline.

**Inputs**: Golden QA dataset, compiled ablation variants.  
**Outputs**: `evaluation/ablation_results.json`, verified production pipeline, final delivery report.  
**Interconnectivity**: Validates the entire AmorphNet system against the problem statement.

---

## 5. Detailed Interconnectivity & Data Flow Matrix

The following matrix defines the exact contracts, schemas, and interfaces connecting every phase of AmorphNet:

| Upstream Phase | Downstream Phase | Data Artifact Transferred | Schema / Format / Protocol | Failure Handling & Throttling |
|---|---|---|---|---|
| **Phase 0** (Config & Clients) | **All Phases** | Settings, Client Singletons, Rate Limiters | Pydantic `Settings`, `AsyncLimiter`, `Client` instances | Fast fail at startup if cloud credentials or connections fail |
| **Phase 1** (Dataset) | **Phase 3 & 14** (Evaluation) | Ground Truth QA Pairs | JSON array of `{question, ground_truth, reference_chunks}` | Stratified sampling for `smoke_eval` mode (10 items) |
| **Phase 4** (Ingestion & Preproc) | **Phase 5** (Chunker) | Structured Pages & Element Maps | List of `LlamaParseResult` (`markdown_content`, `element_type_map`) | `asyncio.Semaphore(3)` throttling; `tenacity` retry on 429 |
| **Phase 5** (Chunker) | **Phase 6** (Metadata Enricher) | Raw Element Chunks | List of `Chunk` objects (`text`, `element_type`, `table_caption`, etc.) | Length verification; hard token cap 768 tokens |
| **Phase 6** (Enricher) | **Phase 7** (Dual Indexing) | Enriched Chunks with Financial Metadata | `Chunk` objects + `financial_metrics`, `fiscal_periods`, `scale` | Batched (6–8 chunks/call) via Gemini `AsyncLimiter(14, 60)` |
| **Phase 7** (Dual Indexing) | **Phase 9** (Hybrid Retrieval) | Cloud Datastores | Qdrant `financial_chunks` collection & Neo4j AuraDB graph | Upsert batches <= 100 points, Semaphore(5); Neo4j constraints |
| **Phase 8** (Query Understanding) | **Phase 9** (Hybrid Retrieval) | Query Representations & Filters | `QueryAnalysis`, dense vector (1024), HyDE vector, BM25 text | Consolidated Gemini call; fall back to verbatim query if parser fails |
| **Phase 9** (Hybrid Retrieval) | **Phase 10** (Re-ranking) | Candidate Chunk Pool | 30 unique `Chunk` objects with fused RRF scores | Dedup by `chunk_id`; graph chunks given baseline rank 21 |
| **Phase 10** (Re-ranking) | **Phase 11** (Context & Gen) | Top-6 High Precision Chunks | List of 6 `Chunk` objects + relevance scores + table boost | Voyage `AsyncClient(max_retries=5)`; fallback to top RRF if rerank fails |
| **Phase 11** (Context & Gen) | **Phase 12 & 13** (Graph & API) | Validated Cited Answer | `AnswerSchema` JSON (`answer`, `citations`, `has_answer`) | Post-validation quote check; strip invalid claims |
| **Phase 12** (LangGraph) | **Phase 13** (FastAPI) | End-to-End Query Execution | Async function call accepting query string -> `AnswerSchema` | 30-second query timeout; exception handling with clean 500 error |
| **Phase 13** (API / Cache) | **End User / Client** | Cached or Generated HTTP Response | REST JSON payload + rate limit headers (`X-RateLimit-*`) | Upstash Redis 1h TTL; SlowAPI HTTP 429 with `Retry-After` |
| **Phase 2 & 12** (Pipelines) | **Phase 14** (Ablation Harness) | Pipeline Outputs for Evaluation | RAGAS evaluation dataframe (`question`, `contexts`, `answer`) | `InMemoryRateLimiter(0.2)` strictly sequential judge calls |

---

## 6. Execution Progress Tracker

Use this master checklist to track AmorphNet implementation progress within Antigravity IDE:

- [x] **Phase 0: Environment, Configuration & Core Infrastructure Setup**
  - [x] 0.1 Virtual Environment & Package Management Setup
  - [x] 0.2 Directory Layout Creation
  - [x] 0.3 Configuration & Secrets Management (`amorphnet/config.py`)
  - [x] 0.4 Shared Cloud Client Singletons & Rate Limiting Guardrails
  - [x] 0.5 Service Connectivity Verification Script (`scripts/check_services.py`)
- [ ] **Phase 1: Golden Evaluation Dataset & Evaluation Harness Foundation**
  - [ ] 1.1 Curate Golden Financial QA Dataset (`evaluation/golden_dataset.json`)
  - [ ] 1.2 Build RAGAS Evaluator Harness (`amorphnet/evaluation/ragas_runner.py`)
  - [ ] 1.3 Build Evaluation Execution Script (`scripts/run_evaluation.py`)
- [ ] **Phase 2: Tier 1 Baseline Pipeline Implementation**
  - [ ] 2.1 Baseline Document Ingestion & Chunking (`amorphnet/ingestion/baseline_ingest.py`)
  - [ ] 2.2 Baseline Vector Indexing (`amorphnet/ingestion/baseline_indexer.py`)
  - [ ] 2.3 Baseline Dense Retrieval (`amorphnet/retrieval/baseline_retriever.py`)
  - [ ] 2.4 Baseline Generation (`amorphnet/generation/baseline_generator.py`)
- [ ] **Phase 3: Tier 1 Baseline Evaluation & Benchmark Floor Establishment**
  - [ ] 3.1 Ingest Benchmark Filings into Baseline Index
  - [ ] 3.2 Execute Baseline Evaluation Run (`--mode smoke`)
  - [ ] 3.3 Document Baseline Performance Floor in `ablation_results.json`
- [ ] **Phase 4: Ingestion — Structure-Aware Parsing & Document Registry**
  - [ ] 4.1 Document Registry (`amorphnet/ingestion/document_registry.py`)
  - [ ] 4.2 PDF Pre-Processing & Rate-Limit Guardrails (`amorphnet/ingestion/pdf_preprocessor.py`)
  - [ ] 4.3 LlamaParse v2 Cloud Client Integration (`amorphnet/ingestion/llamaparse_client.py`)
- [ ] **Phase 5: Chunking Strategy — Heterogeneous Element-Aware Chunker**
  - [ ] 5.1 Standardized Chunk Data Model (`amorphnet/ingestion/chunk_schema.py`)
  - [ ] 5.2 Hierarchical Text Splitter (`amorphnet/ingestion/chunker.py`)
  - [ ] 5.3 Table Chunking Engine (`amorphnet/ingestion/chunker.py`)
  - [ ] 5.4 Chart & Footnote Chunking (`amorphnet/ingestion/chunker.py`)
- [ ] **Phase 6: Metadata Enrichment — Batched Extraction & Breadcrumbs**
  - [ ] 6.1 Batched Financial Entity Enricher (`amorphnet/ingestion/metadata_enricher.py`)
  - [ ] 6.2 Section Hierarchy & Breadcrumb Builder
- [ ] **Phase 7: Dual-Store Indexing — Qdrant Cloud & Neo4j AuraDB**
  - [ ] 7.1 Qdrant Collection Initialization (`amorphnet/ingestion/qdrant_indexer.py`)
  - [ ] 7.2 Batched Embedding & Qdrant Ingestion
  - [ ] 7.3 Neo4j AuraDB Knowledge Graph Construction (`amorphnet/ingestion/neo4j_indexer.py`)
  - [ ] 7.4 Cross-Reference Detection Engine (`amorphnet/ingestion/neo4j_indexer.py`)
- [ ] **Phase 8: Query Understanding, Routing & Query-Side HyDE**
  - [ ] 8.1 Query Analysis Structured Model (`amorphnet/retrieval/query_classifier.py`)
  - [ ] 8.2 Consolidated Query Understanding Node
  - [ ] 8.3 Multi-Representation Embedding Generation (`amorphnet/retrieval/query_expander.py`)
- [ ] **Phase 9: Hybrid Retrieval & Neo4j Graph Expansion**
  - [ ] 9.1 Qdrant 3-Prefetch RRF Query (`amorphnet/retrieval/qdrant_retriever.py`)
  - [ ] 9.2 Neo4j Knowledge Graph Expansion (`amorphnet/retrieval/graph_retriever.py`)
  - [ ] 9.3 Candidate Pool Consolidation & Baseline RRF Assignment
- [ ] **Phase 10: Re-ranking — Voyage rerank-2.5 with Table-Aware Boosting**
  - [ ] 10.1 Voyage rerank-2.5 Integration (`amorphnet/retrieval/reranker.py`)
  - [ ] 10.2 Table-Aware Score Boosting Logic
- [ ] **Phase 11: Context Assembly, Citation Generation & Post-Validation**
  - [ ] 11.1 Standardized Citation Envelope Formatter (`amorphnet/generation/context_assembler.py`)
  - [ ] 11.2 Citation-Enforced Generation (`amorphnet/generation/generator.py`)
  - [ ] 11.3 Post-Generation Citation Validation Gate (`amorphnet/generation/citation_validator.py`)
- [ ] **Phase 12: LangGraph Stateful Orchestration Pipeline**
  - [ ] 12.1 Define Pipeline State Schema (`amorphnet/graph/state.py`)
  - [ ] 12.2 Construct LangGraph Workflow (`amorphnet/graph/langgraph_pipeline.py`)
  - [ ] 12.3 Unit & Integration Testing of Graph Traversal
- [ ] **Phase 13: Production API Layer, Cloud Caching & Multi-Tier Throttling**
  - [ ] 13.1 Upstash Redis Serverless Caching & Task State (`amorphnet/api/cache.py`)
  - [ ] 13.2 Multi-Tier Rate Limiting Middleware (`amorphnet/api/middleware.py`)
  - [ ] 13.3 Circuit Breakers for Cloud Dependencies
  - [ ] 13.4 FastAPI Application Endpoints (`amorphnet/api/main.py`)
- [ ] **Phase 14: Full Pipeline Evaluation, Ablation Benchmark & Final Delivery**
  - [ ] 14.1 Ablation Study Execution Framework (`amorphnet/evaluation/ablation.py`)
  - [ ] 14.2 Benchmark Run & Quota Management (`smoke` vs `full`)
  - [ ] 14.3 Metrics Verification vs Architecture Targets (Faithfulness >= 0.90)
  - [ ] 14.4 Persist Benchmark Report & Update Documentation
