# MAAT: Egyptian Legal Research Assistant

## Project documentation for poster and presentation

**MAAT** is a bilingual (Arabic/English) Egyptian legal research assistant. It uses Retrieval-Augmented Generation (RAG) to find relevant passages from an Egyptian legal corpus and use them as evidence while an AI model drafts a response. The project is designed to make legal research more accessible, structured, and source-aware; it is **not** a substitute for a qualified lawyer or formal legal advice.

> **Presentation positioning:** MAAT is an AI-powered legal research assistant that combines legal-document retrieval with language generation to provide clearer, evidence-grounded answers about Egyptian law.

---

## 1. Problem statement

Egyptian legal texts are extensive, technical, and often difficult for students and non-specialists to search efficiently. A general-purpose chatbot can produce fluent answers, but it may rely on unsupported information or invent legal details. Legal research needs a system that can first locate relevant source material and then explain it carefully.

MAAT addresses this problem by:

- Searching a curated Egyptian legal corpus before answering legal questions.
- Giving the language model retrieved legal passages as its evidence.
- Displaying the legal sources returned for an answer.
- Supporting Arabic and English interactions.
- Encouraging users to verify sources and consult a lawyer before taking action.

### Core research question

**How can RAG improve the usefulness and trustworthiness of an AI assistant for Egyptian legal research?**

---

## 2. Goals and scope

### Main goals

1. Provide a friendly interface for asking Egyptian legal research questions.
2. Ground answers in retrieved legal text rather than relying only on the model's general knowledge.
3. Help users inspect the source laws behind an answer.
4. Support conversational follow-up questions without treating prior conversation as legal evidence.
5. Evaluate retrieval and answer quality with reproducible metrics.

### Current scope

- Egyptian legal corpus only.
- Informational research support, not legal representation or binding advice.
- Arabic and English user interface and query support.
- Account registration, login, conversations, preferences, and document tracking.

### Important limitation

User-uploaded files are currently stored per user but are **not indexed into the shared RAG corpus**. They can be tracked in the interface, but they do not yet influence retrieval or generation.

---

## 3. System overview

MAAT has two major layers:

1. **Application layer:** a Streamlit web interface for authentication, chat, documents, research prompts, preferences, and source display.
2. **RAG layer:** a legal-corpus pipeline that retrieves relevant text chunks and asks a language model to produce an evidence-based answer.

### Current implemented architecture

```mermaid
flowchart LR
    A[Egyptian legal corpus] --> B[Filter target legal categories]
    B --> C[Chunk documents\n500 tokens, 80-token overlap]
    C --> D[Create multilingual embeddings\nmultilingual-e5-small]
    D --> E[(Chroma vector database)]

    U[User question] --> R{Small talk?}
    R -- Yes --> G1[Groq chat model]
    R -- No --> Q[Embed query]
    Q --> E
    E --> V[Top-k vector retrieval]
    V --> P[Prompt with retrieved legal passages]
    P --> G2[Groq chat model]
    G2 --> O[Answer + retrieved source metadata]
```

### Current query flow

1. The user submits a question in the chat interface.
2. The routing component detects simple social conversation, such as greetings or thanks.
3. Small-talk questions are sent directly to the conversational model.
4. Legal questions are embedded with the same multilingual embedding model used for the corpus.
5. Chroma returns the top *k* semantically similar chunks; the user can select 3, 4, or 6 sources in settings.
6. Retrieved passages and their law names are inserted into a guarded prompt.
7. A Groq-hosted chat model generates an answer using those passages as the legal authority.
8. MAAT returns the answer and the metadata for the retrieved sources.

---

## 4. Planned target architecture: Hybrid RAG

The planned enhancement is a **Hybrid RAG architecture using BM25 and dense vector retrieval, followed by a cross-encoder reranker**. This is the architecture that should be featured as the project direction in the poster and presentation.

### Why Hybrid RAG?

Dense vector retrieval is strong at semantic similarity: it can match a question and a legal passage even when they use different wording. However, law often depends on exact language, article numbers, names, and legal terms. BM25 is strong at exact keyword matching. Combining both methods improves the chance that relevant legal evidence appears in the candidate set.

The cross-encoder reranker then reads the question and each candidate chunk together, allowing it to make a more precise relevance decision than either first-stage retriever alone.

### Target Hybrid RAG pipeline

```mermaid
flowchart LR
    A[Legal corpus] --> C[Clean, chunk, and attach metadata]
    C --> B1[BM25 lexical index]
    C --> B2[Embedding model]
    B2 --> V[(Vector index / Chroma)]

    Q[User legal question] --> N[Normalize query]
    N --> L[BM25 retrieves top-N\nexact terms, article numbers]
    N --> E[Embed query]
    E --> S[Vector search retrieves top-N\nsemantic similarity]
    L --> F[Candidate fusion\nRRF or weighted fusion]
    S --> F
    F --> X[Cross-encoder reranker\nscore question + passage pairs]
    X --> K[Top-k evidence passages]
    K --> P[Grounded generation prompt]
    P --> M[LLM response]
    M --> O[Answer, sources, and confidence signals]
```

### Components of the target design

| Component | Role | Why it matters for legal research |
| --- | --- | --- |
| BM25 | Lexical retrieval using term frequency and rarity | Finds exact legal phrases, law names, and article references. |
| Dense retriever | Embedding-based semantic retrieval | Finds related concepts even when wording differs. |
| Candidate fusion | Combines BM25 and dense results | Increases recall by keeping useful candidates from both retrieval methods. |
| Cross-encoder reranker | Scores each question-passage pair jointly | Improves precision by promoting the most directly relevant evidence. |
| Grounded generator | Writes the user-facing response from selected evidence | Produces a readable explanation while keeping sources central. |

### Recommended retrieval sequence

1. Retrieve a broad candidate set from BM25 (for example, top 20).
2. Retrieve a broad candidate set from dense vector search (for example, top 20).
3. Merge and deduplicate results using **Reciprocal Rank Fusion (RRF)** or a weighted normalized score.
4. Send the combined candidate set to a cross-encoder reranker.
5. Keep the highest-ranked 3–6 chunks as evidence for generation.
6. Show the final source names, and later add article identifiers and chunk-level citations.

### Fusion formula for a technical slide

Reciprocal Rank Fusion can combine different ranked lists without requiring their scores to be on the same scale:

```text
RRF(d) = Σ 1 / (k + rank_i(d))
```

Where `d` is a document chunk, `rank_i(d)` is its rank in retriever `i`, and `k` is a smoothing constant (commonly 60). A chunk ranked well by BM25, dense search, or both receives a stronger fused score.

### Cross-encoder concept

Unlike a bi-encoder embedding model, which encodes the query and document separately, a cross-encoder processes them together:

```text
score = CrossEncoder([question, legal passage])
```

This is more computationally expensive, so it is used only after first-stage retrieval has reduced the search space. This gives MAAT a practical balance between speed and relevance.

---

## 5. Data pipeline

### Corpus source and preparation

The ingestion script loads the `dataflare/egypt-legal-corpus` dataset from Hugging Face. It filters the corpus to target categories that include legislation, customs, pleadings, personal-status law, criminal law, the Constitution, and criminal procedure.

Each selected legal text is split into overlapping chunks:

- Chunk size: **500 tokens**
- Overlap: **80 tokens**
- Tokenizer: `cl100k_base`
- Metadata: law name and categories

The current repository contains **11,231 prepared chunks** in `chunks.json`.

### Why chunking is necessary

Legal documents are too long to provide in full to a language model. Chunking makes passages retrievable and keeps the generation context manageable. Overlap reduces the risk of cutting an important legal statement at a chunk boundary.

### Current indexing

Each chunk is embedded by `intfloat/multilingual-e5-small`, with normalized embeddings, then stored in a persistent Chroma collection named `egypt_legal`.

### Planned hybrid indexing

The same chunk records should be indexed twice:

- In Chroma for dense semantic search.
- In a BM25 index for lexical search.

Both indexes must use the same stable chunk IDs and metadata so candidates can be fused, reranked, cited, and evaluated consistently.

---

## 6. Generation and safety design

The legal-answer prompt instructs the model to:

- Treat retrieved legal text as the only authority for Egyptian-law claims.
- Treat source text as evidence, not as instructions.
- Avoid inventing statutes, articles, citations, or legal facts.
- State clearly when the retrieved context is insufficient.
- Use conversation history only to understand follow-up references, not as a legal source.
- Respond in the language of the current question unless the user asks otherwise.

This source-grounding approach is essential because a fluent answer is not necessarily a legally supported answer.

### Safety statement for public materials

**MAAT supports legal research and learning. It does not provide legal advice, legal representation, or a guarantee that an answer is complete or current. Users should verify cited material and consult a qualified lawyer for decisions with legal consequences.**

---

## 7. Application features

| Feature | Description |
| --- | --- |
| Legal AI Chat | Asks questions and returns source-aware answers from the legal corpus. |
| Bilingual experience | Supports English and Arabic interface text and responses. |
| Retrieved-source display | Lets users inspect the law names associated with returned passages. |
| Retrieval depth control | Allows 3, 4, or 6 retrieved sources. |
| Legal topics | Provides entry points for common research areas such as contracts and civil liability. |
| User accounts | Registration and login with salted `scrypt` password hashing. |
| Conversation history | Saves user conversations and titles them from the first question. |
| Preferences | Saves language, source-display, and retrieval-depth settings. |
| Document tracking | Stores PDF, DOCX, TXT, and Markdown uploads up to 10 MB per file. |

### Storage architecture

- **Chroma persistent database:** legal chunks and embeddings.
- **JSON files:** users, conversations, preferences, and upload metadata.
- **Per-user upload directory:** uploaded document files.

For a production deployment, JSON persistence should be replaced with a database such as PostgreSQL, and uploaded documents should be protected with robust access control and secure object storage.

---

## 8. Technology stack

| Layer | Technologies currently used |
| --- | --- |
| User interface | Streamlit, CSS |
| Language models | Groq chat-completions API with configurable fallback models |
| Embeddings | Sentence Transformers, `intfloat/multilingual-e5-small` |
| Vector store | ChromaDB persistent collection |
| Dataset tooling | Hugging Face Datasets |
| Chunking | Tiktoken |
| Evaluation | Custom retrieval scoring and RAGAS-based generation evaluation |
| Authentication | Python `scrypt`, JSON persistence, file locking utilities |
| Runtime | Python, PyTorch |

### Planned additions for Hybrid RAG

- A BM25 library/index (for example, `rank-bm25`, Elasticsearch/OpenSearch, or a custom sparse index).
- A multilingual cross-encoder reranker suitable for Arabic and English legal text.
- Rank fusion implementation, preferably RRF at first.
- Stable chunk IDs, article metadata, and per-stage retrieval logs.

---

## 9. Evaluation methodology and current baseline

MAAT includes scripts to create an evaluation set, run retrieval and generation, score retrieval, score generated answers, and produce reports.

### Evaluation set

- Current evaluation set size: **91 items**
- Retrieval-scored legal items: **82**
- Question types include single-row, multi-row, and abstention/trap cases.

### Retrieval metrics

| Metric | Meaning | Current dense-retrieval baseline |
| --- | --- | ---: |
| Recall@k | Whether the expected law appears in the retrieved top-k results | 0.829 |
| MRR | How highly the expected law is ranked | 0.671 |
| Source-chunk hit | Whether a known evidence chunk is returned | 0.537 |
| Chunking loss rate | Misses caused by chunk boundaries when the law was found | 0.353 |

### Generation metrics

The current saved RAGAS sample is small, so it should be presented as an initial diagnostic rather than a final benchmark.

| Metric | Current recorded result |
| --- | ---: |
| RAGAS sample size | 3 |
| Faithfulness | 1.000 |
| Answer relevancy | 0.975 |
| Context precision | 0.833 |
| Context recall | 1.000 |
| Abstention accuracy (9 trap cases) | 0.889 |

### How Hybrid RAG should be evaluated

Run the same evaluation set for three systems:

1. Dense vector retrieval baseline.
2. BM25-only retrieval.
3. Hybrid retrieval plus cross-encoder reranking.

Compare Recall@k, MRR, source-chunk hit rate, latency, faithfulness, answer relevancy, and abstention accuracy. The success criterion should be an improvement in evidence retrieval and answer support without unacceptable latency.

> Do not claim that Hybrid RAG has already improved the metrics until the BM25 and reranker implementation has been evaluated. The values above describe the current dense-retrieval baseline.

---

## 10. Implementation roadmap

### Phase 1 — Reliable corpus and metadata

- Preserve stable chunk IDs during chunking and indexing.
- Add source URL, document version/date, article number, and article title where available.
- Normalize Arabic text consistently for BM25 while retaining the original display text.
- Check for duplicates and malformed source records.

### Phase 2 — Hybrid retrieval

- Build the BM25 index over the same chunks used by Chroma.
- Retrieve candidates independently from BM25 and dense search.
- Fuse rankings with RRF.
- Log which retriever contributed each candidate.

### Phase 3 — Cross-encoder reranking

- Select a multilingual cross-encoder that handles Arabic and English.
- Rerank only the fused candidate set to control latency.
- Return final top-k chunks with relevance scores.
- Add a minimum-score threshold for abstention or a clarification request.

### Phase 4 — Better citations and user trust

- Display law name, article number, source excerpt, and retrieval/reranker score.
- Link to authoritative source material where licensing and availability allow.
- Clearly separate: supported findings, assumptions, and missing facts.

### Phase 5 — Production readiness

- Move user data from JSON files to PostgreSQL.
- Add role-based access controls, encryption, audit logs, backups, and rate limiting.
- Isolate each user's private document index from the shared legal corpus.
- Establish a corpus-update and legal-review workflow.

---

## 11. Poster-ready content

### Suggested title

**MAAT: A Hybrid RAG Assistant for Source-Grounded Egyptian Legal Research**

### One-sentence abstract

MAAT is a bilingual Egyptian legal research assistant that combines lexical search, semantic retrieval, cross-encoder reranking, and grounded language generation to deliver clearer answers connected to legal evidence.

### Poster sections

1. **Problem** — Legal texts are complex; generic AI can produce unsupported answers.
2. **Solution** — MAAT retrieves legal evidence before generating an explanation.
3. **Architecture** — Show the Hybrid RAG diagram: BM25 + dense retrieval → fusion → cross-encoder reranking → grounded LLM.
4. **Data** — Egyptian legal corpus, 11,231 chunks, chunk metadata.
5. **Evaluation** — Dense baseline: Recall@k 82.9%, MRR 67.1%, source-chunk hit 53.7%.
6. **Impact** — Faster source-oriented legal research for students and the public.
7. **Limitations and ethics** — Informational support only; verify sources and consult a lawyer.
8. **Future work** — Implement and evaluate the hybrid retriever, article-level citations, and secure production storage.

### Poster callout numbers

- **11,231** legal text chunks
- **Arabic + English** interaction
- **82.9%** dense baseline Recall@k
- **3 retrieval depths:** 3, 4, or 6 sources
- **Hybrid RAG target:** BM25 + vector search + cross-encoder reranking

### Short architecture caption

“MAAT combines the exact-term strength of BM25 with the semantic understanding of embeddings. A cross-encoder reranks the combined candidates before the LLM writes a source-grounded answer.”

---

## 12. Presentation slide outline

| Slide | Title | Key message |
| --- | --- | --- |
| 1 | MAAT | Egyptian legal research, grounded in retrieved sources. |
| 2 | The problem | Legal research is difficult; unrestricted AI may hallucinate legal claims. |
| 3 | Our solution | RAG retrieves evidence first, then generates an explanation. |
| 4 | Current prototype | Streamlit + embeddings + Chroma + Groq; show current baseline flow. |
| 5 | Why improve retrieval? | Dense search alone may miss exact legal terminology or article references. |
| 6 | Hybrid RAG architecture | BM25 and vector search run together; fused results are reranked by a cross-encoder. |
| 7 | What each component contributes | BM25 = exact terms; dense retrieval = meaning; reranker = precision. |
| 8 | Data and preparation | Corpus filtering, 500-token chunks, 80-token overlap, metadata. |
| 9 | Product experience | Chat, source display, bilingual UI, accounts, and preferences. |
| 10 | Evaluation baseline | Present existing dense-retrieval metrics and clarify Hybrid RAG is future work. |
| 11 | Safety and limitations | Not legal advice; corpus coverage and source currency matter. |
| 12 | Roadmap | BM25, fusion, reranking, citations, private-document indexing, production storage. |
| 13 | Closing | “Better retrieval leads to more trustworthy legal AI.” |

### Suggested 60-second architecture narration

“When a user asks a legal question, MAAT searches the corpus in two ways. BM25 looks for exact terms such as law names, legal phrases, and article numbers. Dense retrieval looks for passages with similar meaning, even if the wording is different. We merge these candidates and let a cross-encoder compare the question with each passage in detail. The best evidence is then passed to the language model, which produces an answer grounded in the retrieved legal text and shows the relevant sources.”

---

## 13. Honest project status for the presentation

Use this wording to avoid confusing the current prototype with the planned research upgrade:

> “The current MAAT prototype is operational with dense vector retrieval using multilingual embeddings and ChromaDB. The next version upgrades the retrieval stage to Hybrid RAG: BM25 and dense retrieval will be fused, then refined by a cross-encoder reranker. This direction is motivated by the need to handle both exact legal wording and semantic similarity.”

Avoid saying “the project currently uses BM25 and a cross-encoder reranker” until that code has been added, indexed, and evaluated.

---

## 14. Repository map

| File / folder | Responsibility |
| --- | --- |
| `app.py` | Streamlit user interface and application flow. |
| `RAG.py` | Embedding loading, Chroma retrieval, prompts, and Groq generation. |
| `data.py` | Dataset download/filtering and chunk generation. |
| `embed.py` | Embedding and Chroma indexing script. |
| `chunks.json` | Prepared legal chunks with source metadata. |
| `chroma_db/` | Persistent Chroma vector database. |
| `services/` | Authentication and user-data services. |
| `utils/chat_routing.py` | Small-talk and creator-question routing. |
| `generate_eval_set.py` | Builds the evaluation set. |
| `run_eval.py` | Executes retrieval and answer generation on evaluation questions. |
| `score_retrieval.py` | Computes retrieval metrics. |
| `score_generation.py` | Computes generation and abstention metrics. |
| `eval_report.md` | Detailed evaluation report. |
| `tests/` | Tests for response behavior and chat routing. |

---

## 15. Local development notes

### Main prerequisites

- Python environment with the packages in `requirements.txt`.
- A `GROQ_API_KEY` configured in `.env` for response generation.
- A populated `chroma_db` collection named `egypt_legal`.
- A locally available embedding model, or `RAG_EMBED_LOCAL_FILES_ONLY=false` to permit downloading it.

### Typical workflow

```powershell
# 1. Create chunks from the legal corpus
python data.py

# 2. Embed and index chunks in Chroma
python embed.py

# 3. Launch the interface
streamlit run app.py
```

### Evaluation workflow

```powershell
# Run retrieval and generation on the evaluation set
python run_eval.py --eval-set eval_set.json --output eval_results.json --k 4

# Score retrieval
python score_retrieval.py --results eval_results.json --output retrieval_scores.json

# Score generated answers
python score_generation.py
```

Run commands from the project environment and use the installed virtual environment where applicable.

---

## 16. Key takeaways

- MAAT is a source-oriented Egyptian legal research assistant, not a replacement for legal counsel.
- The implemented prototype uses multilingual dense embeddings, Chroma vector search, guarded prompts, and Groq generation.
- The target innovation is Hybrid RAG: **BM25 + dense retrieval + rank fusion + cross-encoder reranking**.
- The hybrid design is especially appropriate for legal research because it combines exact terminology with semantic understanding.
- Evaluation, transparent citations, safe abstention, and current authoritative sources are as important as a fluent answer.
