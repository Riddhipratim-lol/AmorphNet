# AmorphNet — Architecture

> **Purpose-built financial RAG pipeline for annual reports and SEC filings**

---

## Table of Contents

1. [Problem Statement & Design Principles](#1-problem-statement--design-principles)
2. [System Overview](#2-system-overview)
3. [Tech Stack — Canonical Versions](#3-tech-stack--canonical-versions)
4. [Phase 0 — Document Ingestion & Pre-Processing](#4-phase-0--document-ingestion--pre-processing)
5. [Phase 1 — Structure-Aware Parsing (LlamaParse)](#5-phase-1--structure-aware-parsing-llamaparse)
6. [Phase 2 — Chunking Strategy](#6-phase-2--chunking-strategy)
7. [Phase 3 — Metadata Enrichment](#7-phase-3--metadata-enrichment)
8. [Phase 4 — Dual-Store Indexing (Qdrant + Neo4j)](#8-phase-4--dual-store-indexing-qdrant--neo4j)
9. [Phase 5 — Query Understanding & Routing](#9-phase-5--query-understanding--routing)
10. [Phase 6 — Hybrid Retrieval (BM25 + Dense + Graph)](#10-phase-6--hybrid-retrieval-bm25--dense--graph)
11. [Phase 7 — Reciprocal Rank Fusion (RRF)](#11-phase-7--reciprocal-rank-fusion-rrf)
12. [Phase 8 — Re-ranking (Voyage rerank-2.5)](#12-phase-8--re-ranking-voyage-rerank-2.5)
13. [Phase 9 — Context Assembly & Citation Tracking](#13-phase-9--context-assembly--citation-tracking)
14. [Phase 10 — Generation (Gemini 3.5 Flash-Lite + Citation Enforcement)](#14-phase-10--generation-gemini-3.5-flash-lite--citation-enforcement)
15. [Phase 11 — Evaluation (RAGAS)](#15-phase-11--evaluation-ragas)
16. [Phase 12 — API Layer (FastAPI)](#16-phase-12--api-layer-fastapi)
17. [Project Directory Layout](#17-project-directory-layout)
18. [Data Flow — End-to-End Diagram](#18-data-flow--end-to-end-diagram)
19. [Configuration & Secrets Management](#19-configuration--secrets-management)
20. [Key Design Decisions & Rationale](#20-key-design-decisions--rationale)

---

## 1. Problem Statement & Design Principles

Annual reports (10-K, 10-Q, 20-F) and financial filings are structurally dense. They are not prose — they are multi-modal, multi-layout documents containing:

- **Tabular data** (income statements, balance sheets, cash flow tables) — structure destroyed by naive PDF text extraction
- **Multi-column layouts** — column text is concatenated out of order by page-level parsers
- **Charts and graphs** — silently dropped as images by most parsers
- **Footnotes and endnotes** — stripped of their referential anchors when parsed as plain text
- **Cross-references** — phrases like "as described in Note 7" point to sections hundreds of pages away

Naive RAG pipelines that treat everything as flat text will:
- Fail to answer table-grounded questions ("What was net revenue in FY2023?")
- Miss chart-encoded data (revenue trend visuals)
- Hallucinate footnote context (confident wrong answers)
- Silently drop cross-referenced information

**AmorphNet's design principles:**

| Principle | Implication |
|---|---|
| Structure preservation | Parse tables as markdown tables, not linearized strings |
| Multi-modality | Charts are described; their data is captured |
| Referential integrity | Footnotes carry their parent anchor as metadata |
| Grounded generation | Every claim in the answer must carry a source citation |
| Precision over recall | Re-rank aggressively; better to retrieve fewer, accurate chunks |
| Evaluable | Every pipeline stage has measurable metrics via RAGAS |

---

## 2. System Overview

AmorphNet has three major sub-systems:

```
+---------------------------------------------------------------------+
|                       INGESTION PIPELINE                           |
|  PDF Input -> LlamaParse -> Chunker -> Metadata -> Qdrant + Neo4j  |
+---------------------------------------------------------------------+
                                |
                                v
+---------------------------------------------------------------------+
|                       QUERY PIPELINE                               |
|  User Query -> Query Router -> BM25+Dense+Graph -> RRF -> Rerank   |
|             -> Context Assembly -> Gemini -> Cited Answer           |
+---------------------------------------------------------------------+
                                |
                                v
+---------------------------------------------------------------------+
|                       EVALUATION HARNESS                           |
|  QA Dataset -> RAGAS Metrics -> Score Dashboard                    |
+---------------------------------------------------------------------+
```

All three sub-systems are orchestrated by LangGraph and exposed via FastAPI.

---

## 3. Tech Stack — Canonical Versions

> **Deployment model**: All datastores and AI services run **entirely on managed cloud infrastructure via API**. No local databases, no Docker containers, no self-hosted services. The only process running locally (or on your deployment host) is the Python FastAPI application itself.

| Layer | Technology | Version / Driver / Model | Role |
|---|---|---|---|
| Document Parsing | LlamaParse Cloud API | `llama-cloud-services >=0.6.94` (API v2) | Structure-aware PDF parsing with multimodal output |
| Embeddings | `voyage-finance-2` | `voyageai >=0.5.0` via Voyage AI API (cloud) | Finance-domain dense embeddings (1024 dim) |
| Sparse Vectorizer | BM25 (Qdrant Cloud native inference) | Qdrant Cloud v1.14+ | Keyword-based sparse vectors — generated server-side on Qdrant Cloud |
| Vector Store | **Qdrant Cloud** | `qdrant-client >=1.19.0` (managed cloud API) | Dense + sparse named vectors, hybrid Query API |
| Graph Store | **Neo4j AuraDB** | `neo4j >=6.3.0` / AuraDB v5.26+ (managed cloud API) | Document/section/entity relationship graph |
| Fusion | RRF (built into Qdrant Query API) | Qdrant Cloud v1.14+ | Reciprocal Rank Fusion of BM25 + dense prefetches |
| Re-ranking | `rerank-2.5` | `voyageai >=0.5.0` via Voyage AI API (cloud) | Cross-encoder precision re-ranking — 32k context with instruction-following |
| Orchestration | LangChain + LangGraph | `langchain >=1.4.0`, `langgraph >=1.2.11` | Pipeline DAG + stateful agent workflows |
| Generation | `gemini-3.5-flash-lite` | `google-genai >=2.23.0` via Google GenAI API (cloud) | Citation-grounded answer generation |
| Evaluation | RAGAS | `ragas >=0.4.3` | Automated RAG quality metrics |
| API & Schema | FastAPI + Pydantic | `fastapi >=0.141.1`, `pydantic >=2.13.5` | REST API serving query and ingestion endpoints + schemas |
| Async Runtime | asyncio + uvicorn | `uvicorn >=0.52.4`, Python 3.12+ | Fully async pipeline execution |
| Caching | **Upstash Redis** (Serverless) | `upstash-redis >=1.8.0` / `redis >=8.1.0` | Cloud-hosted serverless Redis — query cache + ingestion job state |

> **Note on `voyage-finance-2` vs `voyage-4-large`**: Voyage AI's latest general-purpose series is `voyage-4-large` (32k context, 1024 dim). However, `voyage-finance-2` remains the recommended choice for **financial domain retrieval** because it was fine-tuned on financial corpora and produces superior in-domain recall on SEC filings, earnings reports, and accounting text. Use `voyage-finance-2` for all chunk and query embeddings in this project.

> **Note on `rerank-2.5` vs older `rerank-2`**: `rerank-2.5` is Voyage AI's current recommended reranker (as of 2025–2026), succeeding `rerank-2`. It adds instruction-following support and multilingual capability at a 32k context window, versus `rerank-2`'s 16k. Use `rerank-2.5` for all re-ranking steps.

> **Note on Gemini model**: `gemini-3.5-flash-lite` (API model ID: `gemini-3.5-flash-lite`) via the official `google-genai` SDK is the recommended generation model for this project. It is Google's flagship Flash-Lite generation, offering optimal latency, ultra-low token cost, and robust structured JSON compliance for citation-grounded financial RAG.

---

## 4. Phase 0 — Document Ingestion & Pre-Processing

### 4.1 Input Handling

Documents enter the system through an **ingestion endpoint** (`POST /ingest`). Supported input formats:

- PDF (primary — annual reports, 10-K, 10-Q, 20-F, prospectuses)
- DOCX (converted to PDF server-side before parsing)
- XLSX (processed separately via tabular pipeline, not through LlamaParse)

Each document is registered in a **Document Registry** table (SQLite or PostgreSQL depending on deployment scale) with:

```
document_id     UUID (primary key)
filename        string
company_name    string (extracted or provided)
fiscal_year     int (extracted or provided)
filing_type     string  ("10-K", "10-Q", "20-F", etc.)
page_count      int
ingestion_date  datetime
status          enum ("pending", "parsing", "indexed", "failed")
```

### 4.2 Deduplication Gate

Before parsing, compute a **SHA-256 hash** of the raw PDF bytes. If the hash already exists in the Document Registry, skip re-ingestion. This prevents re-parsing the same annual report twice and avoids creating duplicate vectors in Qdrant.

### 4.3 Pre-Processing Checks

Before sending to LlamaParse:
- Validate the PDF is not password-protected
- Detect if the PDF is a scanned image-only document (using PyMuPDF's `page.get_images()`) — flag this as `scan_mode=True` so LlamaParse uses its OCR mode
- Extract basic document metadata (title, author, creation date) from PDF XMP metadata using PyMuPDF
- Count total pages; documents over 600 pages are split into 200-page batches for parallel async parsing

---

## 5. Phase 1 — Structure-Aware Parsing (LlamaParse)

### 5.1 Why LlamaParse

LlamaParse's agentic mode uses vision-language models internally to understand PDF layout before extracting content. This is fundamentally different from text-extraction tools (pdfminer, pymupdf text extraction, unstructured) which operate on the character stream layer and have no concept of visual layout.

LlamaParse v2 (invoked via the modern `llama-cloud-services` SDK, succeeding the deprecated `llama-parse` package) provides:
- **Markdown output** with preserved table structure (tables become proper `| col | col |` markdown, not linearized strings)
- **Per-page bounding box metadata** — every extracted element knows its page number and pixel coordinates
- **Chart detection and description** — charts are identified and a natural language description is generated (e.g., "Bar chart showing revenue by segment for FY2022-2024")
- **Multi-column layout handling** — columns are read in the correct left-to-right, top-to-bottom order
- **Footnote anchoring** — footnotes are returned with a reference to their parent anchor text

### 5.2 Parse Configuration

```yaml
parse_mode: "parse"
output_format: "markdown"
language: "en"
premium_mode: true
skip_diagonal_text: true
take_screenshot: false
invalidate_cache: false
page_separator: "---PAGE {page_num}---"
```

**Financial-domain parsing instructions** (system prompt override for LlamaParse):

```
You are parsing a financial document (annual report, 10-K, or similar SEC filing).
- Preserve ALL tables exactly as structured, rendered as markdown tables.
- For every table, include a one-line caption above it describing what the table shows.
- Detect charts and graphs; describe their axes, legend, and key data points in a structured paragraph.
- Preserve footnote markers (e.g., "(1)", "a)") and include the full footnote text immediately after the section it annotates.
- For multi-column layouts, read left column fully before right column.
- Do not merge separate line items in financial tables into single rows.
- Preserve all currency symbols, unit annotations, and scaling notes (e.g., "in millions").
```

### 5.3 Parse Output Structure

Each parsed document returns a list of `LlamaParseResult` objects, one per page. Each page result contains:
- `page_number` (int)
- `markdown_content` (str) — the full parsed text for that page
- `element_type_map` (dict) — maps character offset ranges to element types: "text", "table", "chart_description", "footnote"
- `raw_images` (list) — base64-encoded page screenshots (used for chart elements if vision mode is enabled)

---

## 6. Phase 2 — Chunking Strategy

This is the most critical design decision in the pipeline. Financial documents require a **heterogeneous chunking strategy** — one size does not fit all.

### 6.1 Element-Type-Aware Chunking

After parsing, each page's markdown is split by element type detected in Phase 1:

#### 6.1.1 Text Chunks (Narrative Sections)

Narrative prose (MD&A, Risk Factors, Business Overview) is chunked using a **hierarchical splitter**:

- **Primary split**: Section headers (detected via markdown `##` and `###` headers from LlamaParse output)
- **Secondary split**: Semantic sentence-boundary splitting within sections, targeting `512 tokens` per chunk (using the voyage-finance-2 tokenizer for accurate token counting)
- **Overlap**: `50 token` sliding window overlap between consecutive text chunks within the same section — this preserves context across sentence boundaries
- **Maximum chunk size**: `768 tokens` (hard cap; anything larger will miss nuance during retrieval)

Rationale for 512 tokens: `voyage-finance-2` has a 32k context window but embedding quality degrades at very large chunk sizes. 512 tokens is empirically optimal for dense retrieval on financial text.

#### 6.1.2 Table Chunks

Tables require special treatment. A naive approach would embed the entire markdown table as one chunk — but large balance sheets can be 1000+ tokens, and embedding that as a single vector loses the ability to retrieve specific cells or rows.

**Table chunking rules:**

1. **Small tables** (<= 300 tokens): Kept as a single chunk. The entire markdown table plus its caption is one chunk.
2. **Medium tables** (301-800 tokens): Split **row-group-wise** — group rows by semantic category (e.g., all "Revenue" rows together, all "Expenses" rows together) based on indentation and row header text. Each row group + the full header row = one chunk.
3. **Large tables** (> 800 tokens): Split into chunks of <= 10 rows each, with the **table header always prepended** to each chunk. This ensures every chunk contains the column context.

Additionally, every table chunk preserves the **table caption** generated during Phase 1 parsing in its metadata (`table_caption`). This column and context summary is indexed in the Qdrant payload to support high-level table filtering without requiring redundant per-table LLM summary calls.

#### 6.1.3 Chart Description Chunks

Chart descriptions (generated by LlamaParse's vision model) are kept as single chunks. They are tagged with `element_type: "chart"` and their source image is stored as a reference in the metadata. Chart chunks are typically 100-300 tokens.

#### 6.1.4 Footnote Chunks

Footnotes are chunked individually (one chunk per footnote). Each footnote chunk carries `parent_anchor` metadata pointing to the text chunk that contains the footnote reference marker. This enables a **footnote retrieval cascade**: when a text chunk is retrieved, its associated footnotes can be automatically fetched from Qdrant using a payload filter on `parent_anchor`.

### 6.2 Chunk Metadata Schema

Every chunk (regardless of type) carries a standardized metadata payload:

```python
{
    "chunk_id":          str,   # UUID
    "document_id":       str,   # Parent document UUID
    "company_name":      str,   # e.g., "Apple Inc."
    "ticker":            str,   # e.g., "AAPL"
    "filing_type":       str,   # "10-K", "10-Q", "20-F"
    "fiscal_year":       int,   # e.g., 2024
    "fiscal_quarter":    str,   # "Q1", "Q2", "Q3", "Q4", "Annual"
    "page_number":       int,   # Source page in original PDF
    "section_header":    str,   # Nearest parent section heading
    "element_type":      str,   # "text" | "table" | "chart" | "footnote"
    "table_caption":     str,   # Only for table chunks; empty otherwise
    "parent_anchor":     str,   # Only for footnote chunks; the marker text
    "chunk_index":       int,   # Position of chunk within its section
    "token_count":       int,   # Token count using voyage tokenizer
    "char_offset_start": int,   # Character offset in page markdown
    "char_offset_end":   int,
}
```

---

## 7. Phase 3 — Metadata Enrichment

Before indexing, each chunk passes through a **metadata enrichment pipeline** that adds derived fields using lightweight, structured LLM calls.

To strictly respect API rate limits (e.g., free tier 15 RPM / 500 RPD) while preserving semantic precision without brittle regex, enrichment is executed using **chunk batching** and client-side token-bucket rate limiting.

### 7.1 Batched Financial Entity Extraction

Instead of one call per chunk, chunks are grouped into **batches of 6–8 consecutive chunks** and processed in a single structured Gemini 3.5 Flash-Lite call using `response_schema`:

```python
from pydantic import BaseModel
from google.genai import types

class ChunkMetadata(BaseModel):
    chunk_id: str
    financial_metrics: list[str]      # e.g., ["revenue", "net_income", "EPS", "EBITDA"]
    fiscal_periods: list[str]         # e.g., ["FY2024", "Q3 2024"]
    currency_and_scale: str | None    # e.g., "USD in millions"
    is_comparative: bool              # True if comparing two periods (e.g., "up 12% YoY")

class BatchEnrichmentResponse(BaseModel):
    enriched_chunks: list[ChunkMetadata]
```

**Key Advantages:**
1. **90% Call Reduction**: A 200-chunk filing requires only ~25–30 API calls instead of 200+ calls, completing in under 2.5 minutes at 12 RPM and consuming ~5% of daily quota.
2. **Context Coherence**: Consecutive chunks provide surrounding paragraph context (e.g., table headers, unit scale notes), which prevents hallucination of missing units.
3. **Rate Limiting**: Calls are throttled using `aiolimiter.AsyncLimiter(max_rate=14, time_period=60)` with exponential backoff on `429 RESOURCE_EXHAUSTED`.

These extracted fields are stored as structured metadata payload attributes and indexed for **filtered retrieval** (e.g., payload filter `financial_metrics CONTAINS "revenue"`).

### 7.2 Query-Side HyDE (Decoupled from Ingestion)

The pipeline employs **Query-Side HyDE** (in [Section 9.2](#92-query-expansion)) rather than index-time question pre-computation:
- **Zero Ingestion Overhead**: No hypothetical questions are generated or embedded during ingestion, saving hundreds of LLM calls and eliminating secondary named vector storage (`hyp_question`) in Qdrant.
- **Canonical Formulation**: At query time, Gemini generates a single hypothetical financial passage that is embedded with `voyage-finance-2` and searched directly against the primary `dense` chunk vector. This achieves identical vocabulary-bridging recall without bloat.

### 7.3 Section Hierarchy Extraction

Parse each chunk's `section_header` metadata to build a **breadcrumb path**:
- `"Part II > Item 8 > Note 3 — Revenue Recognition"`

This path is stored as metadata and used in the Neo4j graph construction in Phase 4.

---

## 8. Phase 4 — Dual-Store Indexing (Qdrant + Neo4j)

### 8.1 Qdrant — Vector + Sparse Index

#### 8.1.1 Collection Design

Create a single Qdrant collection: `financial_chunks` with **named dense and sparse vector configurations**:

```python
from qdrant_client import models

vectors_config = {
    "dense": models.VectorParams(
        size=1024,                  # voyage-finance-2 output dim
        distance=models.Distance.COSINE,
        hnsw_config=models.HnswConfigDiff(m=16, ef_construct=200)
    )
}

sparse_vectors_config = {
    "sparse": models.SparseVectorParams(
        modifier=models.Modifier.IDF   # Server-side BM25 IDF modifier
    )
}
```

#### 8.1.2 Sparse Vector Generation (BM25)

Qdrant Cloud's native BM25 Inference API generates sparse vectors **server-side** from the chunk text — no local tokenizer or local model required. The sparse vector represents IDF-weighted term frequencies across the vocabulary. The `modifier: "idf"` configuration enables proper BM25 scoring instead of raw term frequency.

Because this project uses **Qdrant Cloud**, the BM25 computation is offloaded entirely to the cloud cluster. There is no dependency on local FastEmbed or SPLADE binaries.

#### 8.1.3 Payload Indexing

The following metadata fields are indexed in Qdrant for fast filtered retrieval:

```python
payload_indexes = [
    ("company_name",   "keyword"),
    ("ticker",         "keyword"),
    ("filing_type",    "keyword"),
    ("fiscal_year",    "integer"),
    ("fiscal_quarter", "keyword"),
    ("element_type",   "keyword"),
    ("page_number",    "integer"),
    ("section_header", "text"),     # Full-text payload index for section name search
]
```

#### 8.1.4 Ingestion Batch Size

Upload chunks to Qdrant in batches of **100 points** using the async Python client. For large documents (600+ pages), run parallel batch uploads using `asyncio.gather()` with a concurrency limit of 5 batch uploads at a time.

### 8.2 Neo4j — Document Knowledge Graph

The knowledge graph serves two purposes:
1. **Structural traversal**: Given a retrieved chunk, find neighboring sections, parent sections, and cross-referenced sections
2. **Entity-based lookup**: Given a financial entity mention (e.g., "Note 7 — Income Taxes"), retrieve all chunks associated with that entity

#### 8.2.1 Node Types

```cypher
(:Document {
    document_id: string,
    company_name: string,
    filing_type: string,
    fiscal_year: int
})

(:Section {
    section_id: string,
    document_id: string,
    header_text: string,
    breadcrumb: string,
    page_number: int,
    element_type: string
})

(:Chunk {
    chunk_id: string,
    section_id: string,
    element_type: string,
    page_number: int,
    token_count: int
})

(:FinancialEntity {
    name: string,
    entity_type: string   // "metric", "period", "segment", "geography"
})
```

#### 8.2.2 Relationship Types

```cypher
(:Document)-[:HAS_SECTION]->(:Section)
(:Section)-[:HAS_CHILD_SECTION]->(:Section)
(:Section)-[:HAS_CHUNK]->(:Chunk)
(:Chunk)-[:MENTIONS]->(:FinancialEntity)
(:Chunk)-[:REFERENCES]->(:Chunk)       // Cross-references (e.g., "See Note 7")
(:Chunk)-[:FOOTNOTE_OF]->(:Chunk)      // Footnotes pointing to parent
(:Section)-[:NEXT_SECTION]->(:Section) // Sequential ordering
```

#### 8.2.3 Cross-Reference Detection

During ingestion, scan each text chunk for patterns matching cross-references:
- `"See Note \d+"`, `"as described in Note \d+"`
- `"Refer to [A-Z][a-z]+ \d+"`, `"discussed in Item \d+"`
- `"as shown in [Tt]able \d+"`

When detected, create `(:Chunk)-[:REFERENCES]->(:Chunk)` edges. These edges are used at query time to expand the retrieved context: if a retrieved chunk references "Note 7", automatically fetch Note 7's chunks from Qdrant via the `chunk_id` stored on the graph node.

---

## 9. Phase 5 — Query Understanding & Routing

Before retrieval, every incoming user query passes through a **query understanding** step implemented as a LangGraph node.

### 9.1 Query Classification

Classify the query into one of four types:

| Query Type | Description | Example | Retrieval Strategy |
|---|---|---|---|
| `factual_numeric` | Asks for a specific number from a table | "What was Apple's net revenue in FY2024?" | Dense + sparse; prefer table chunks |
| `trend_comparative` | Asks for change over time or comparison | "How did revenue grow from FY2022 to FY2024?" | Dense; multi-year document range |
| `semantic_risk` | Asks about qualitative risk disclosures | "What are the key cybersecurity risks disclosed?" | Dense; prefer text chunks |
| `cross_reference` | References a specific section or note | "What does Note 3 say about lease obligations?" | Section-filtered retrieval + graph traversal |

Classification is done using a Gemini 3.5 Flash-Lite call with structured output (Pydantic model). To conserve rate limits, classification, expansion, and filter extraction can be executed in a single consolidated `QueryAnalysis` LLM call.

### 9.2 Query Expansion & Query-Side HyDE

For the query, generate **three representations**:
1. **Original query** (verbatim): Embedded with `voyage-finance-2` as `dense_query_vector`.
2. **Keyword-optimized variant** (for BM25): e.g., `"net revenue total sales FY2024 fiscal year 2024"`, converted to a sparse vector for Qdrant's BM25 index.
3. **Hypothetical answer fragment** (Query-Side HyDE): e.g., `"Apple's net revenue for fiscal year 2024 was $XXX billion, representing a Y% increase..."`, embedded with `voyage-finance-2` as `hyde_query_vector`.

Both the original query and the hypothetical answer are embedded into the same 1024-dimensional space and matched against Qdrant's primary `"dense"` vector field.

### 9.3 Metadata Filter Extraction

Extract structured filters from the query:
- **Company name / ticker** (if mentioned): `company_name = "Apple Inc."` or `ticker = "AAPL"`
- **Fiscal year** (if mentioned): `fiscal_year = 2024`
- **Filing type** (if mentioned): `filing_type = "10-K"`
- **Element type preference** (derived from query type): For `factual_numeric`, prefer `element_type = "table"`; for `semantic_risk`, prefer `element_type = "text"`

These filters are applied as Qdrant payload filters to constrain the retrieval space, dramatically improving precision.

---

## 10. Phase 6 — Hybrid Retrieval (BM25 + Dense + Graph)

### 10.1 Qdrant Hybrid Query

Execute a single Qdrant Query API call using three prefetches fused server-side with RRF:

```python
query_result = qdrant_client.query_points(
    collection_name="financial_chunks",
    prefetch=[
        # Prefetch 1: BM25 sparse retrieval on keyword-optimized query
        Prefetch(
            query=SparseVector(indices=bm25_sparse.indices, values=bm25_sparse.values),
            using="sparse",
            limit=40,
            filter=payload_filter
        ),
        # Prefetch 2: Dense retrieval on original query embedding
        Prefetch(
            query=dense_query_vector,
            using="dense",
            limit=40,
            filter=payload_filter
        ),
        # Prefetch 3: Query-side HyDE retrieval on hypothetical answer embedding against dense chunks
        Prefetch(
            query=hyde_query_vector,
            using="dense",
            limit=20,
            filter=payload_filter
        ),
    ],
    query=RrfQuery(rrf=Rrf(k=60)),    # k=60 is the standard RRF constant
    limit=20,
    with_payload=True,
    with_vectors=False
)
```

The 40/40/20 prefetch limits are asymmetric because dense retrieval is the primary signal for semantic understanding, BM25 is equally important for financial number lookups, and HyDE is supplemental. After RRF fusion, we take the top 20.

### 10.2 Graph-Augmented Retrieval

After the Qdrant query returns the top 20 chunks, execute a **graph expansion** step in Neo4j:

For each retrieved chunk:
1. Fetch all `[:FOOTNOTE_OF]` footnote chunks (limit: 3 per chunk)
2. Follow `[:REFERENCES]` edges to retrieve directly cross-referenced sections (limit: 2 per chunk)
3. Fetch 1 preceding and 1 following chunk via `[:NEXT_SECTION]` edges for context continuity

This graph expansion adds up to ~10 additional chunks. Deduplicate the combined set by `chunk_id`. The result is a **candidate pool** of up to 30 unique chunks.

---

## 11. Phase 7 — Reciprocal Rank Fusion (RRF)

RRF is applied **inside Qdrant** across the three prefetch queries (dense, sparse, HyDE). The formula Qdrant uses:

```
score(d) = sum { 1 / (k + r_d) }  for each ranking r containing document d
```

where `k=60` and `r_d` is the zero-based rank position in each individual result list.

The graph-expansion chunks that were not in the Qdrant top-20 are appended to the pool with a **rank position of 21** (last place) in each ranking list, giving them a baseline RRF score. They are included because they have structural relevance even if they did not score highly in vector search.

Final pool going into re-ranking: **30 chunks**, ranked by RRF score.

---

## 12. Phase 8 — Re-ranking (Voyage rerank-2.5)

### 12.1 Why Re-rank After RRF

RRF fuses rank positions from multiple retrieval signals but it does not perform **semantic cross-attention** between the query and each individual chunk. A cross-encoder (re-ranker) reads both the query and the chunk text together and scores their relevance — this is far more precise than embedding-based similarity or rank-position fusion.

### 12.2 Voyage rerank-2.5 Configuration

```python
rerank_result = voyage_client.rerank(
    query=original_user_query,
    documents=[chunk.text for chunk in candidate_pool],
    model="rerank-2.5",   # Current recommended model; 32k context window
    top_k=6
)
```

**Why `rerank-2.5` over `rerank-2`?** `rerank-2.5` is Voyage AI's current production-recommended reranker. It doubles the context window from 16k to 32k tokens (critical for long financial tables), adds instruction-following capability, and maintains multilingual support — all at the same API price point.

**Why 6?** Six chunks of up to 768 tokens each = ~4,608 tokens of context. With Gemini 3.5 Flash-Lite's context window, this is sufficient for comprehensive answers while keeping generation focused. More context leads to attention dilution and increased hallucination risk.

### 12.3 Table-Aware Re-ranking Adjustment

For `factual_numeric` queries, apply a **score boost** of `+0.15` to any chunks with `element_type = "table"` before calling the re-ranker. This prevents the re-ranker from deprioritizing dense tabular text in favor of flowing narrative text.

```python
final_score = rerank_score * (1.15 if chunk.metadata["element_type"] == "table" else 1.0)
```

---

## 13. Phase 9 — Context Assembly & Citation Tracking

### 13.1 Citation Envelope

Each of the top-6 re-ranked chunks is wrapped in a **citation envelope** before being passed to the generation model:

```
[SOURCE 1]
Company: Apple Inc. | Filing: 10-K | Fiscal Year: 2024 | Page: 42
Section: Part II > Item 8 > Consolidated Statements of Operations
Element Type: table

| | 2024 | 2023 | 2022 |
|---|---|---|---|
| Net sales | $391,035 | $383,285 | $394,328 |
...
[END SOURCE 1]

[SOURCE 2]
Company: Apple Inc. | Filing: 10-K | Fiscal Year: 2024 | Page: 44
Section: Part II > Item 8 > Note 1 -- Revenue Recognition
Element Type: text

Revenue is recognized when control of promised goods or services is transferred to customers...
[END SOURCE 2]
```

This format embeds all citation metadata directly in the context string, making it unambiguous for the generation model to attribute claims.

### 13.2 System Prompt Design for Citation Enforcement

The generation system prompt enforces citation on every factual claim:

```
You are a financial analyst assistant answering questions about company annual reports and SEC filings.

STRICT RULES:
1. Every factual claim, number, or data point in your answer MUST be cited with [SOURCE N] where N is the source number provided in the context.
2. If the information to answer the question is NOT present in the provided sources, respond exactly: "The provided documents do not contain sufficient information to answer this question."
3. Do NOT invent, infer, or extrapolate numbers not explicitly stated in the sources.
4. If numbers are in different fiscal years, clearly state which year each number corresponds to.
5. When citing tables, specify the row and column you are reading from.
6. Do not combine or summarize numbers from multiple sources without explicitly noting you are doing so.
```

---

## 14. Phase 10 — Generation (Gemini 3.5 Flash-Lite + Citation Enforcement)

### 14.1 LangGraph Generation Node

The generation step is implemented as a LangGraph node within a stateful graph. The state object carries:

```python
class QueryState(TypedDict):
    original_query:    str
    query_type:        str
    expanded_queries:  list[str]
    filters:           dict
    candidate_chunks:  list[Chunk]
    reranked_chunks:   list[Chunk]
    context_string:    str
    answer:            str
    citations:         list[Citation]
    confidence_score:  float
```

### 14.2 Generation Call

AmorphNet uses Google's modern `google-genai` SDK (`google-genai >=2.23.0`):

```python
from google import genai
from google.genai import types

client = genai.Client()

response = client.models.generate_content(
    model="gemini-3.5-flash-lite",   # Current recommended Flash-Lite model
    contents=f"Context:\n{context_string}\n\nQuestion: {original_query}",
    config=types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.1,
        max_output_tokens=1024,
        response_mime_type="application/json",
        response_schema=AnswerSchema,
    ),
)
```

**Structured output schema** (`AnswerSchema`):

```python
class Citation(BaseModel):
    source_number: int
    company_name:  str
    filing_type:   str
    fiscal_year:   int
    page_number:   int
    quote:         str    # Exact text fragment from the source

class AnswerSchema(BaseModel):
    answer:          str
    citations:       list[Citation]
    has_answer:      bool
    confidence:      float
    caveats:         list[str]
```

### 14.3 Post-Generation Validation

After generation, run a **citation validation pass**:
1. Parse the `answer` string for all `[SOURCE N]` markers
2. Verify that each N has a corresponding entry in `citations`
3. Verify that the `quote` in each citation exists verbatim in the corresponding source chunk
4. If any citation is invalid (hallucinated source number, or quote does not match), log a warning and strip the uncited claim or re-generate with a stricter prompt

This validation is a lightweight string-matching step, not an LLM call.

---

## 15. Phase 11 — Evaluation (RAGAS)

### 15.1 Evaluation Dataset Construction

Build a **golden QA dataset** of 200 questions by:
1. Manually curating 50 questions from 5-6 representative annual reports (10 questions per document)
2. Generating 150 additional synthetic questions using Gemini 3.5 Pro (`gemini-3.5-pro`) over the parsed documents
3. Distributing across query types: 30% `factual_numeric`, 25% `trend_comparative`, 25% `semantic_risk`, 20% `cross_reference`

Each QA pair has:
```python
{
    "question":         str,
    "ground_truth":     str,
    "reference_chunks": list[str]   # chunk_ids of the gold-standard source chunks
}
```

### 15.2 RAGAS Metrics

| Metric | What It Measures | Target |
|---|---|---|
| `ContextPrecision` | What fraction of retrieved chunks are actually relevant | >= 0.75 |
| `ContextRecall` | What fraction of relevant chunks were retrieved | >= 0.80 |
| `Faithfulness` | Does the answer only contain claims supported by retrieved context? | >= 0.90 |
| `AnswerRelevance` | How well does the answer address the question? | >= 0.80 |
| `AnswerCorrectness` | Is the answer factually correct vs the ground truth? | >= 0.75 |

`Faithfulness >= 0.90` is the primary hard target. For financial data, hallucination is a showstopper metric.

### 15.3 Evaluation Pipeline Invocation

Modern RAGAS (v0.4.3) uses class-based metric instances:

```python
from ragas import evaluate
from ragas.metrics import (
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    Faithfulness,
    ResponseRelevancy,
    AnswerCorrectness,
)

result = evaluate(
    dataset=eval_dataset,
    metrics=[
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
        Faithfulness(),
        ResponseRelevancy(),
        AnswerCorrectness(),
    ],
    llm=gemini_flash_lite_wrapper,
    embeddings=voyage_finance_2_wrapper,
)
```

### 15.4 Ablation Studies

| Variant | Description |
|---|---|
| `baseline` | No hybrid retrieval; dense only; no re-ranking |
| `+bm25` | Dense + BM25 hybrid; no re-ranking |
| `+rerank` | Dense + BM25 + re-ranking; no graph |
| `+graph` | Dense + BM25 + re-ranking + graph expansion |
| `+hyde` | Full pipeline with HyDE query expansion |
| `full` | Complete AmorphNet pipeline (all components) |

Results are written to `evaluation/ablation_results.json`.

---

## 16. Phase 12 — API Layer (FastAPI)

### 16.1 Endpoints

```
POST  /ingest
      Body: { file: <binary PDF>, company_name: str, fiscal_year: int, filing_type: str }
      Response: { document_id: str, status: "queued" }

GET   /ingest/{document_id}/status
      Response: { document_id: str, status: str, chunks_indexed: int }

POST  /query
      Body: { question: str, filters?: { company_name?, fiscal_year?, filing_type? } }
      Response: AnswerSchema

POST  /evaluate
      Body: { qa_pairs?: list, use_golden_dataset?: bool }
      Response: { scores: RAGASScoreReport }

GET   /health
      Response: { status: "ok", qdrant_cloud: "ok", neo4j_aura: "ok", upstash_redis: "ok" }
```

### 16.2 Request Handling Architecture

- Ingestion requests (`POST /ingest`) are **async background tasks** — the endpoint returns immediately with a `document_id` and the parsing/indexing pipeline runs in the background. Job progress is tracked in Upstash Redis via `HSET task:{task_id}` entries.
- Query requests (`POST /query`) run synchronously within a 30-second timeout; all internal async calls (Qdrant Cloud, Neo4j AuraDB, Voyage AI, Gemini) use `asyncio.gather()` for parallelism where possible
- **Upstash Redis** caches query results for 1 hour keyed by `SHA256(question + filters)` — repeated identical queries are served from cache at near-zero cost and < 15ms latency

### 16.3 Rate Limiting & Error Handling

- Rate limit the `/query` endpoint to 10 requests/minute per API key
- Implement circuit breakers for Voyage AI API and Qdrant Cloud; degrade gracefully if cloud services are unreachable
- All LlamaParse API calls use the async polling mode (submit job -> poll every 15s -> fetch result) to handle large documents without blocking threads

---

## 17. Project Directory Layout

```
amorphnet/
├── pyproject.toml
├── .python-version          (3.12.x)
├── .env.example
│
├── amorphnet/
│   ├── __init__.py
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── document_registry.py
│   │   ├── llamaparse_client.py
│   │   ├── chunker.py
│   │   ├── metadata_enricher.py
│   │   ├── qdrant_indexer.py
│   │   └── neo4j_indexer.py
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── query_classifier.py
│   │   ├── query_expander.py
│   │   ├── filter_extractor.py
│   │   ├── qdrant_retriever.py
│   │   ├── graph_retriever.py
│   │   └── reranker.py
│   │
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── context_assembler.py
│   │   ├── generator.py
│   │   └── citation_validator.py
│   │
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── langgraph_pipeline.py
│   │   └── state.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── ragas_runner.py
│   │   ├── dataset_builder.py
│   │   └── ablation.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── routes/
│   │   │   ├── ingest.py
│   │   │   ├── query.py
│   │   │   └── evaluate.py
│   │   └── middleware.py
│   │
│   └── config.py
│
├── evaluation/
│   ├── golden_dataset.json
│   └── ablation_results.json
│
├── scripts/
│   ├── ingest_batch.py
│   └── run_evaluation.py
│
└── tests/
    ├── unit/
    │   ├── test_chunker.py
    │   ├── test_metadata_enricher.py
    │   └── test_citation_validator.py
    └── integration/
        ├── test_ingestion_pipeline.py
        └── test_query_pipeline.py
```

---

## 18. Data Flow — End-to-End Diagram

### Ingestion Flow

```
PDF File
    |
    v
[Pre-Processing: hash check, scan detection, page count]
    |
    v
[LlamaParse v2: markdown + element type map per page]
    |
    v
[Heterogeneous Chunker: text / table / chart / footnote chunks]
    |
    v
[Metadata Enricher: batched financial entity extraction (6-8 chunks/call)]
    |
    +----------------------------------------------+
    v                                              v
[Qdrant: upsert dense + sparse vectors            [Neo4j: create Document,
 + payload metadata]                               Section, Chunk nodes + edges]
```

### Query Flow

```
User Question
    |
    v
[Query Classifier -> query_type enum]
    |
    v
[Query Expander -> 3 query variants + embedding]
    |
    v
[Filter Extractor -> payload filter dict]
    |
    v
[Qdrant Hybrid Query: 3 prefetches (sparse/dense/hyp) -> RRF -> top 20]
    |
    v
[Neo4j Graph Expansion: footnotes + cross-refs + neighbors -> +10 chunks]
    |
    v
[Dedup -> ~30 candidate chunks]
    |
    v
[Voyage rerank-2.5 -> top 6 chunks]
    |
    v
[Context Assembler -> citation envelopes]
    |
    v
[Gemini 3.5 Flash-Lite -> AnswerSchema with inline citations]
    |
    v
[Citation Validator -> post-generation check]
    |
    v
[Upstash Redis cache write (cloud, TTL=1h)]
    |
    v
AnswerSchema response to API caller
```

---

## 19. Configuration & Secrets Management

All configuration is managed via a Pydantic `Settings` class reading from environment variables. All services are accessed via cloud APIs — **no local services, no Docker required**.

```
# .env.example

# LlamaParse Cloud API
LLAMA_CLOUD_API_KEY=llx-...

# Voyage AI (embeddings + reranking)
VOYAGE_API_KEY=pa-...

# Qdrant Cloud (managed vector DB)
QDRANT_URL=https://<cluster-id>.us-east4-0.gcp.cloud.qdrant.io
QDRANT_API_KEY=...
QDRANT_COLLECTION=financial_chunks

# Neo4j AuraDB (managed knowledge graph)
NEO4J_URI=neo4j+s://<aura-db-id>.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=...

# Google AI (Gemini 3.5 Flash-Lite)
GOOGLE_API_KEY=AIza...

# Upstash Redis (serverless cloud cache — no local Redis)
UPSTASH_REDIS_REST_URL=https://<your-db>.upstash.io
UPSTASH_REDIS_REST_TOKEN=...
# Standard redis-py compatible URL (also supported):
# REDIS_URL=rediss://default:<token>@<your-db>.upstash.io:6379

# Application
LOG_LEVEL=INFO
MAX_CONCURRENT_INGESTIONS=3
QUERY_TIMEOUT_SECONDS=30
CACHE_TTL_SECONDS=3600
```

Secrets are **never** committed to version control. Use a `.env` file locally (gitignored) and environment variable injection on your deployment platform (Render, Railway, Google Cloud Run, etc.). A secrets manager like HashiCorp Vault or Doppler is recommended for production.

---

## 20. Key Design Decisions & Rationale

### Why not use a simpler single-vector approach?

Single-vector dense retrieval fails on financial documents because:
- Keyword matches (BM25) are critical for number lookups: "net revenue", "FY2024", "Note 7" are exact-match queries that dense embeddings handle poorly
- Semantic similarity alone cannot distinguish "revenue increased" from "revenue decreased" in some embedding spaces
- Query-side HyDE expansion handles the vocabulary mismatch problem when user questions use different terminology than the source document

### Why batch metadata enrichment (6–8 chunks/call) instead of per-chunk calls?

Financial filings generate hundreds of chunks per document. Calling an LLM once per chunk creates severe API rate limit bottlenecks (e.g., hitting 15 RPM / 500 RPD free-tier limits in minutes) and isolates each chunk from surrounding context. Grouping 6–8 consecutive chunks into a single structured extraction call:
1. **Reduces API calls by ~90%** (e.g., from ~200 calls to ~25 calls for a 100-page 10-K), enabling reliable ingestion under strict rate limits.
2. **Preserves context continuity**, allowing the model to accurately associate parent table captions, currency scale notes ("$ in millions"), and fiscal periods spanning adjacent paragraphs.
3. **Decouples HyDE to query time**, generating a single hypothetical passage during query expansion rather than hundreds of unneeded pre-computed questions during indexing.

### Why Qdrant over Pinecone or Weaviate for this use case?

Qdrant's native hybrid Query API (dense + sparse in a single call with server-side RRF fusion) is unique among vector databases. Pinecone does not support BM25 sparse vectors natively. Weaviate supports hybrid search but does not provide the same level of control over named vector fields and prefetch weighting. Qdrant also supports payload filtering on numerical fields (fiscal year range queries) which is essential for financial data.

### Why Neo4j in addition to Qdrant?

Vector search is fundamentally a nearest-neighbor problem — it cannot answer "give me the section two levels up from this chunk" or "give me the chunk that Note 7 cross-references". The knowledge graph enables structural and relational retrieval that vector search cannot provide. Together, they form a **hybrid retrieval architecture** that covers both semantic similarity and document structure.

### Why voyage-finance-2 over general-purpose embeddings?

Domain-specific embeddings consistently outperform general embeddings on in-domain retrieval tasks. `voyage-finance-2` was trained on financial corpora and understands the semantic relationships between financial terms. For example, it knows that "net revenue", "total sales", "top line", and "revenue" are semantically related in a financial context, whereas `text-embedding-3-large` may not capture these domain-specific synonymies as well.

### Why temperature 0.1 for generation?

Financial answers must be **deterministic and precise**. Higher temperatures introduce creative paraphrasing which can alter the meaning of numerical claims. Temperature 0.1 (near-zero but not zero) prevents the model from always selecting the single highest-probability token while keeping answers highly factual.

### Why 6 final chunks after re-ranking?

This is the precision-recall trade-off calibration point. Testing at 4 chunks: recall too low for multi-part questions. Testing at 10 chunks: answer quality drops (attention dilution, more irrelevant content leads to more caveats). 6 chunks at 512-768 tokens each provides ~4000 tokens of focused context — sufficient for complex multi-part financial questions while maintaining answer precision.

### Why validate citations post-generation?

Structured output schemas (JSON mode) reduce but do not eliminate hallucinated citations. A model may correctly produce a structured `Citation` object but invent the `quote` field. The post-generation validation catches this class of failure without requiring another LLM call — it is a fast string-matching operation that runs in milliseconds.

---

*End of AmorphNet Architecture Document*
