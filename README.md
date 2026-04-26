# learn-rag

A minimal **Retrieval-Augmented Generation (RAG)** API built with [Bun](https://bun.com), [Fastify](https://fastify.dev/), and [Ollama](https://ollama.com/) running locally.

It lets you ingest raw text, embed it into vectors, store them in memory, then answer questions by retrieving the most relevant chunks and generating a response via a local LLM.

## How it works

```
POST /ingest  →  chunk text  →  embed chunks (nomic-embed-text)  →  store vectors in memory
POST /query   →  embed question  →  cosine similarity search  →  generate answer (llama3)
```

## Prerequisites

- [Bun](https://bun.com) v1.3+
- [Ollama](https://ollama.com/) running locally on `http://localhost:11434`
- Ollama models pulled:
  ```bash
  ollama pull nomic-embed-text
  ollama pull llama3
  ```

## Install & run

```bash
bun install
bun start        # starts the server on http://localhost:3000
```

## API

### `POST /ingest`
Chunk and embed a body of text, storing vectors in memory.

```json
{ "text": "your document text here" }
```

Response: `{ "chunks": 4 }`

---

### `POST /query`
Ask a question against the ingested content.

```json
{ "question": "Give me a gist of important points" }
```

Response: `{ "answer": "...", "docs": ["..."] }`

---

### `GET /healthcheck`
Returns `{ "status": "ok" }`.

## Project structure

```
server/
  index.ts              # Fastify app setup & graceful shutdown
  routes/
    ingest.ts           # POST /ingest route
    query.ts            # POST /query route
    test.http           # HTTP test file (REST Client)
  services/
    chunk.ts            # Splits text into word-based chunks (default 500 words)
    embed.ts            # Calls Ollama embeddings API (nomic-embed-text)
    retrieve.ts         # In-memory vector store with cosine similarity search
    generate.ts         # Calls Ollama generate API (llama3)
    index.ts            # Re-exports all services
```

> **Note:** Vectors are stored in memory only — they are lost on server restart.
