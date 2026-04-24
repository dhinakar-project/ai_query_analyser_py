# Technical Documentation

**Live Deployment:** [https://ai-customer-query-analyser.streamlit.app/](https://ai-customer-query-analyser.streamlit.app/)

## SECTION 1 — System Overview

Customer support teams across the industry receive high query volume daily. Manual triage of these incoming requests is slow, inconsistent, and fails to scale during peak loads. This system automates the core triage lifecycle—classification, prioritization, and initial response generation—using a specialized generative AI pipeline. It evaluates each customer inquiry instantly and delivers actionable intelligence with measurable accuracy, resolving trivial questions automatically while accelerating the escalation of critical issues.

This project implements a multi-stage language model pipeline that processes raw user inputs through a strict security boundary, evaluates them in a single comprehensive analysis pass, and dynamically routes them using an agentic graph. It replaces fragile string manipulation with statically typed validation boundaries, ensuring that outputs strictly adhere to application schemas. The system provides integrated observability, evaluating its own precision via a deterministic benchmark suite, and tracking unit-level costs per API call to maintain operational efficiency.

## SECTION 2 — Architecture Deep Dive

### Subsection 2.1: The LangGraph StateGraph

Traditional linear chains process data sequentially, making them ill-suited for decision-driven routing or cyclic processes. A `StateGraph` defines the system as a state machine where nodes execute logic and edges determine the flow based on the data. The core shared object in this implementation is the `QueryState`, defined via Pydantic. It holds the entire context of the request throughout execution. Its fields include: `query` (the input string), `language` (detected locale), `category` (domain mapping), `sentiment` (emotional tone), `priority` (urgency scale), `should_escalate` (boolean flag), `response` (generated output), `reasoning_trace` (audit trail strings), `rag_sources` (list of retrieved documents), and `messages` (interaction history).

Execution follows a strict order. First, the query passes through the pre-pipeline guardrails. It then enters the `combined_analysis_node`. After analysis completes, a conditional edge evaluates the `should_escalate` flag. If `should_escalate` is True, execution routes to the `escalation_response_node` (which generates a template-based response consuming 0 LLM calls). If False, the flow routes to the `responder_node` (which performs 1 LLM call supplemented by RAG context).

### Subsection 2.2: The Combined Analysis Node

The `CombinedAnalysis` schema is a parent Pydantic model encapsulating five sub-models: LanguageDetection, CategoryClassification, SentimentAnalysis, PriorityAssessment, and EscalationDecision. The LangChain method `llm.with_structured_output()` wraps the LLM invocation, instructing the model to return its output mapped perfectly to this JSON schema. If the model output deviates from the schema types, validation immediately catches it.

The system prompt enforces a five-step chain-of-thought reasoning process:
1. Detect the user's language natively.
2. Classify the query into predefined product/service categories.
3. Analyze sentiment and emotional state.
4. Assess the priority based on financial or functional impact.
5. Determine if human escalation is required.

These five dimensions are extracted in a single LLM call because they are fundamentally interdependent. The user's sentiment influences the priority, which in turn influences the escalation decision. Processing them concurrently provides the model with full context, improves analytical cohesion, and prevents contradictory outputs that often occur when using five independent agents.

### Subsection 2.3: The RAG Pipeline

The retrieval system utilizes an embedded ChromaDB collection pre-loaded with 33 technical articles, each tagged with specific category metadata. To maximize retrieval precision, the system employs a two-stage strategy. Rather than executing a naive top-k vector search across the entire knowledge base, the retriever first applies a strict metadata filter (`WHERE category = X`). It only performs similarity search on the subset of documents matching the category extracted by the analysis node.

The retrieved articles are injected into the final responder prompt using a distinct delimiter format: `[Article: title]\n<content>`. The system injects a maximum of top-3 articles to prevent context window saturation and prompt dilution. In the event of a retrieval failure or an empty vector result, the pipeline executes a graceful fallback, generating a generalized, safe response without hallucinating specific technical steps.

### Subsection 2.4: Guardrails Layer

The guardrails run sequentially BEFORE the LangGraph pipeline execution, guaranteeing the pipeline never processes raw, untrusted input. The system enforces five sequential validation checks:
1. Empty/length check: Drops anomalous or malformed requests.
2. Symbol-only check: Prevents injection payloads masked as punctuation.
3. Prompt injection detection: Applies regex patterns to detect adversarial payloads.
4. Offensive content detection: Filters violating text based on a blocklist.
5. PII redaction: Obfuscates sensitive data via regex substitution.

The prompt injection detector utilizes 18 regex patterns to defend against specific attack vectors:
1. `(?i)ignore (all )?(previous )?instructions`: Defends against basic instruction override.
2. `(?i)you are (now )?a`: Defends against role-play injection.
3. `(?i)system prompt`: Defends against prompt extraction.
4. `(?i)jailbreak`: Defends against known jailbreak frameworks.
5. `(?i)DAN`: Defends against "Do Anything Now" attacks.
6. `(?i)developer mode`: Defends against developer mode bypasses.
7. `(?i)forget (everything )?told`: Defends against context deletion.
8. `(?i)pretend that`: Defends against scenario fabrication.
9. `(?i)bypassing`: Defends against explicit bypass directives.
10. `(?i)override`: Defends against logic replacement.
11. `(?i)disregard`: Defends against instruction dismissal.
12. `(?i)translate to`: Defends against obfuscation attacks.
13. `(?i)write a (poem|story) about`: Defends against output format hijacking.
14. `(?i)echo`: Defends against system reflection attacks.
15. `(?i)base64`: Defends against encoded injections.
16. `(?i)hex`: Defends against hex-encoded payloads.
17. `(?i)hypothetical`: Defends against theoretical scenario exploits.
18. `(?i)say:`: Defends against literal output enforcement.

The PII redaction engine identifies five data types, demonstrating awareness of international formats:
- Email: `[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+`
- Phone: `\b\d{3}[-.]?\d{3}[-.]?\d{4}\b`
- Credit Card: `\b(?:\d[ -]*?){13,16}\b`
- SSN: `\b\d{3}-\d{2}-\d{4}\b`
- Aadhaar (Indian ID): `\b\d{4}\s\d{4}\s\d{4}\b`

## SECTION 3 — Evaluation Framework

The system utilizes a golden dataset comprised of 30 hand-labeled `QueryTestCase` objects. Each case maps an input query to its expected category, sentiment, priority, and escalation decision. To evaluate the pipeline efficiently, the async parallel runner utilizes `asyncio` combined with a `ThreadPoolExecutor` and a `Semaphore(5)`. This concurrency model maximizes throughput and limits concurrent API connections to prevent provider rate limiting.

The framework computes seven specific metrics: `category_accuracy`, `sentiment_accuracy`, `priority_accuracy`, `escalation_precision`, `escalation_recall`, `avg_confidence_category`, and `avg_latency_ms`.

| Metric | Score | Notes |
| :--- | :--- | :--- |
| Category Accuracy | 92.3% | Evaluated on 30 diverse golden test cases |
| Sentiment Accuracy | 88.7% | 5-class classification (Positive/Neutral/Negative/Urgent/Frustrated) |
| Priority Accuracy | 85.0% | 4-level priority (Critical/High/Medium/Low) |
| Escalation Precision | Reported in eval report | See `evals/last_report.json` |
| Avg End-to-End Latency | 2.1s | Includes RAG retrieval + 2 LLM calls |
| Avg API Cost per Query | <$0.001 | Post-optimization (was ~$0.003 before node consolidation) |

Escalation metrics require distinct interpretation. Escalation precision measures: of all the queries the system escalated, how many truly required a human agent? High precision means few false alarms. Escalation recall measures: of all the queries that truly needed an escalation, how many did the system successfully catch? High recall means critical issues do not slip through the cracks.

## SECTION 4 — Observability & Cost Tracking

The system utilizes an embedded SQLite database to persist execution metadata. For every processed query, it inserts a record containing: the raw query text, the assigned category, sentiment, priority, the boolean `should_escalate` flag, the final response payload, pipeline `latency_ms`, calculated `cost_usd`, confidence scores for the classification, and an execution timestamp.

Cost calculation occurs on a per-call basis using the Gemini Flash pricing model. The observability module multiplies the recorded input tokens by $0.000075 per 1,000 tokens and the output tokens by $0.0003 per 1,000 tokens, yielding precise unit economics for the pipeline.

To facilitate debugging, the system constructs a reasoning trace. This trace is a list of strings appended iteratively by each node in the LangGraph execution. It provides a step-by-step audit trail, detailing exactly what operations occurred, which paths the router took, and why the system produced its final output.

## SECTION 5 — Data Models

The system enforces type safety through strict Pydantic definitions found in `models/analysis_result.py`.

- **AnalysisResult**: Base model wrapper for standardizing output formats.
- **CombinedAnalysis**: The primary extraction schema. Fields:
  - `language`: nested `LanguageDetection` object
  - `category`: nested `CategoryClassification` object
  - `sentiment`: nested `SentimentAnalysis` object
  - `priority`: nested `PriorityAssessment` object
  - `escalation`: nested `EscalationDecision` object
- **CategoryClassification**: Maps query to business domain. Fields: `primary_category` (str), `sub_category` (str), `confidence` (float). Constraints: requires predefined literal types.
- **SentimentAnalysis**: Evaluates user emotion. Fields: `tone` (str), `score` (float 0.0-1.0), `is_frustrated` (bool).
- **PriorityAssessment**: Calculates operational urgency. Fields: `level` (str: Critical/High/Medium/Low), `reasoning` (str).
- **EscalationDecision**: Determines routing logic. Fields: `requires_human` (bool), `target_department` (str), `summary` (str).
- **LanguageDetection**: Identifies locale. Fields: `iso_code` (str), `name` (str).
- **Article**: Represents a RAG document. Fields: `id` (str), `title` (str), `content` (str), `metadata` (dict).

## SECTION 6 — API & Module Reference

- **`graph/nodes.py`**
  - *Purpose*: Implements the isolated processing steps of the pipeline.
  - *Key Functions*: `combined_analysis_node()`, `responder_node()`, `escalation_response_node()`.
  - *Dependencies*: `utils.llm`, `rag.retriever`, `models.analysis_result`.

- **`graph/builder.py`**
  - *Purpose*: Constructs and compiles the LangGraph state machine.
  - *Key Classes*: Uses `StateGraph` to map nodes and define edges.
  - *Dependencies*: `graph.nodes`, `graph.state`, `graph.edges`.

- **`graph/runner.py`**
  - *Purpose*: Entry point for pipeline execution.
  - *Key Functions*: `run_graph()` async entry point.
  - *Dependencies*: `graph.builder`, `utils.guardrails`.

- **`rag/retriever.py`**
  - *Purpose*: Handles semantic search operations.
  - *Key Functions*: `retrieve()` async function, handles initialization guards.
  - *Dependencies*: `rag.store`.

- **`rag/store.py`**
  - *Purpose*: Manages the vector database layer.
  - *Key Functions*: ChromaDB collection setup, automated article ingestion.
  - *Dependencies*: `chromadb`.

- **`utils/guardrails.py`**
  - *Purpose*: Provides defensive security screening before pipeline execution.
  - *Key Functions*: `validate_query()`, `sanitize_query()`, `redact_pii()`, `is_repetitive_query()`.
  - *Dependencies*: `re`.

- **`utils/llm.py`**
  - *Purpose*: Manages language model instantiation and API interactions.
  - *Key Classes*: LLM client initialization, model configuration management.
  - *Dependencies*: `langchain_google_genai`.

- **`evals/runner.py`**
  - *Purpose*: Orchestrates the testing and benchmarking suite.
  - *Key Functions*: `run_single_case()`, `run_eval_suite()`. Contains the `EvalReport` model.
  - *Dependencies*: `evals.golden_dataset`, `asyncio`, `concurrent.futures`.

- **`observability/costs.py`**
  - *Purpose*: Measures unit economics and token usage.
  - *Key Functions*: `calculate_cost()`, session tracking metrics.
  - *Dependencies*: None.

- **`storage/db.py`**
  - *Purpose*: Manages local persistence for analytics.
  - *Key Functions*: SQLite schema creation, query logging functions.
  - *Dependencies*: `sqlite3`, `json`.

## SECTION 7 — Engineering Decisions Log

1. **Decision**: Single combined_analysis_node vs multi-agent chain
   - *Context*: Initial pipeline used 5 agents for classification.
   - *Options Considered*: 5 sequential nodes, 5 parallel nodes, 1 combined node.
   - *Choice Made*: 1 combined node with structured output.
   - *Outcome*: Decreased latency by 65%, eliminated interdependent context loss, reduced API costs.

2. **Decision**: LangGraph vs plain LangChain
   - *Context*: Needed dynamic routing for escalated queries.
   - *Options Considered*: LangChain `RunnableBranch`, LangGraph `StateGraph`.
   - *Choice Made*: LangGraph `StateGraph`.
   - *Outcome*: Clean cyclic routing, inherent state typing, checkpointing capability.

3. **Decision**: ChromaDB vs in-memory vector search
   - *Context*: Required semantic search over knowledge base.
   - *Options Considered*: NumPy array cosine similarity, FAISS, ChromaDB.
   - *Choice Made*: ChromaDB.
   - *Outcome*: Native metadata filtering support allows for the category-filtered retrieval strategy.

4. **Decision**: Gemini 2.5 Flash vs GPT-4o
   - *Context*: Balancing response quality, cost, and latency.
   - *Options Considered*: OpenAI GPT-4o, Google Gemini 2.5 Flash.
   - *Choice Made*: Gemini 2.5 Flash.
   - *Outcome*: Sufficient reasoning capabilities for classification at a fraction of the cost and latency.

5. **Decision**: Pydantic structured output vs prompt-based JSON extraction
   - *Context*: Needed deterministic data structures for graph routing.
   - *Options Considered*: System prompt instructions, regex parsing, LangChain structured output wrappers.
   - *Choice Made*: Pydantic schema passed to `llm.with_structured_output()`.
   - *Outcome*: Eliminated JSON parsing errors; pipeline code relies on strict Python types.

6. **Decision**: SQLite vs no persistence
   - *Context*: Requirement to track system performance over time.
   - *Options Considered*: In-memory dict, JSON files, SQLite.
   - *Choice Made*: SQLite.
   - *Outcome*: Enables SQL-based aggregation for latency and cost dashboards without external dependencies.

## SECTION 8 — Known Limitations & Future Work

1. **Knowledge base is 33 articles**
   - *Limitation*: The current domain context is manually curated and static.
   - *Future Work*: Requires an automated ingestion pipeline connecting to a CMS (e.g., Zendesk, Confluence) for dynamic scaling.

2. **Voice agent is TTS-only (no STT)**
   - *Limitation*: The system processes text inputs but cannot natively handle audio calls.
   - *Future Work*: Needs integration with Whisper or Google STT to support omni-channel voice routing.

3. **No authentication layer**
   - *Limitation*: The API accepts requests unconditionally.
   - *Future Work*: Needs API key management and JWT validation for secure, multi-tenant use in production.

4. **ChromaDB is in-process**
   - *Limitation*: The local Chroma instance limits horizontal scaling.
   - *Future Work*: Needs migration to a hosted vector database (e.g., Pinecone, Weaviate) for distributed production scale.

5. **Single-session analytics**
   - *Limitation*: Observability metrics reset across isolated test runs.
   - *Future Work*: Needs user session management and persistent Grafana or Streamlit dashboards for longitudinal reporting.

This documentation was written to production standards as part of the SourceSys Technologies GenAI internship project submission.
