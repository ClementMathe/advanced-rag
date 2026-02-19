# Step 8: Production Deployment — Theory

## Table of Contents
1. [Why Deployment Matters](#1-why-deployment-matters)
2. [The Web Server: FastAPI + Uvicorn](#2-the-web-server-fastapi--uvicorn)
3. [Application Architecture: The Factory Pattern](#3-application-architecture-the-factory-pattern)
4. [Configuration Management: Environment Variables](#4-configuration-management-environment-variables)
5. [Request Lifecycle: From HTTP to Answer](#5-request-lifecycle-from-http-to-answer)
6. [Async vs Sync: The Executor Bridge](#6-async-vs-sync-the-executor-bridge)
7. [Streaming: Server-Sent Events (SSE)](#7-streaming-server-sent-events-sse)
8. [Semantic Caching with Redis](#8-semantic-caching-with-redis)
9. [Circuit Breaker Pattern](#9-circuit-breaker-pattern)
10. [Authentication & Middleware](#10-authentication--middleware)
11. [Observability: Metrics, Health Checks, Logging](#11-observability-metrics-health-checks-logging)
12. [Containerization: Docker & Docker Compose](#12-containerization-docker--docker-compose)
13. [GPU Constraints & Single-Worker Design](#13-gpu-constraints--single-worker-design)
14. [Key Takeaways](#14-key-takeaways)

---

## 1. Why Deployment Matters

### The Gap Between "It Works" and "You Can Use It"

After Steps 1-7, our RAG pipeline works perfectly : if you write a Python script, import the classes, and call `pipeline.query("your question")`. But that is only useful to the developer who built it. Nobody else can use the system.

**Deployment** bridges this gap. It wraps your Python code in a web server so that:
- Any computer on the network can send a question and get an answer
- No Python knowledge is needed, just an HTTP request (like visiting a URL)
- The system runs 24/7 without someone manually starting a script
- Multiple users can use it at the same time
- You can monitor if it's healthy, fast, and not burning money

### What Does "Production" Mean?

"Production" means the software is serving real users (not just developers testing it). Production systems need properties that prototypes don't:

| Concern | Prototype (Steps 1-7) | Production (Step 8) |
|---------|----------------------|---------------------|
| **Access** | Python import | HTTP API (any language, any device) |
| **Availability** | Manual start | Auto-restart on crash, health checks |
| **Observability** | Print statements | Structured logs, metrics dashboards |
| **Security** | None | API keys, request validation |
| **Cost control** | Unlimited | Circuit breaker, budget limits |
| **Performance** | Recompute every time | Semantic cache, streaming |
| **Isolation** | Runs on your machine | Docker container (runs anywhere) |

### The Deployment Stack

Our production deployment has four layers:

```
┌──────────────────────────────────────────────────────┐
│                    Grafana (port 3000)               │  Visualization
│               Dashboards & Alerts                    │
├──────────────────────────────────────────────────────┤
│                  Prometheus (port 9090)              │  Metric Storage
│              Scrapes /metrics every 15s              │
├──────────────────────────────────────────────────────┤
│                   Redis (port 6379)                  │  Semantic Cache
│              256 MB, LRU eviction                    │
├──────────────────────────────────────────────────────┤
│                RAG API (port 8000)                   │  Application
│   FastAPI + Uvicorn + Pipeline + GPU (CUDA)          │
└──────────────────────────────────────────────────────┘
```

Each layer runs in its own Docker container. They communicate over an internal network. Only the API port (8000), Prometheus (9090), and Grafana (3000) are exposed to the outside.

---

## 2. The Web Server: FastAPI + Uvicorn

### What Is a Web Server?

A web server is a program that listens for incoming network requests and sends back responses. When you type a URL in your browser, your browser sends an **HTTP request** to a web server, which processes it and sends back an **HTTP response** (usually HTML, JSON, or an image).

```
Client (browser, curl, Python)          Server (our API)
       │                                      │
       │──── HTTP POST /query ───────────────>│
       │     Body: {"query": "Who is...?"}    │
       │                                      │── run pipeline
       │                                      │── format response
       │<─── HTTP 200 OK ────────────────────│
       │     Body: {"answer": "...", ...}     │
       │                                      │
```

### Why FastAPI?

FastAPI is a Python web framework. There are many frameworks (Flask, Django, Express.js, Spring), but FastAPI is particularly well-suited for ML APIs because:

1. **Async by default**: Can handle multiple requests concurrently without blocking
2. **Pydantic validation**: Automatically validates incoming JSON against type-safe schemas
3. **Auto-generated docs**: Creates an interactive API explorer at `/docs` (Swagger UI)
4. **Type hints**: Python type annotations become runtime validation and documentation
5. **Performance**: One of the fastest Python frameworks (comparable to Node.js/Go for I/O-bound tasks)

Example of what Pydantic validation gives you for free:

```python
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=512)
    top_k: int = Field(default=3, ge=1, le=20)

# If someone sends: {"query": "ab"}
# FastAPI auto-responds: 422 Unprocessable Entity
# {"detail": [{"msg": "String should have at least 3 characters"}]}

# If someone sends: {"query": "Who is Beyonce?", "top_k": 100}
# FastAPI auto-responds: 422 Unprocessable Entity
# {"detail": [{"msg": "Input should be less than or equal to 20"}]}
```

Without FastAPI, you would write this validation manually for every endpoint.

### What Is Uvicorn?

FastAPI defines what to do with requests, but it doesn't know how to listen on a network port. **Uvicorn** is the actual server process that:
1. Opens a socket on port 8000
2. Accepts incoming TCP connections
3. Parses raw HTTP bytes into Python objects
4. Passes the request to FastAPI
5. Takes FastAPI's response and sends it back as HTTP bytes

The relationship:

```
Internet → [TCP port 8000] → Uvicorn (ASGI server) → FastAPI (application)
                                  ↑                       ↑
                            Handles networking       Handles logic
```

**ASGI** (Asynchronous Server Gateway Interface) is the protocol between Uvicorn and FastAPI. It's a standard so you can swap servers (Uvicorn, Hypercorn, Daphne) without changing your app.

### Workers

A **worker** is an operating system process running one copy of the application. More workers = more concurrent requests. But for GPU-bound ML models:

```
1 worker  = 1 copy of the model in VRAM  ≈ 2.5 GB (Qwen 3B, 4-bit)
2 workers = 2 copies of the model in VRAM ≈ 5.0 GB → OOM on 6 GB GPU!
```

This is why we use `--workers 1`. The constraint is VRAM, not CPU.

---

## 3. Application Architecture: The Factory Pattern

### The Problem with Global State

A naive approach creates the app at module import time:

```python
# BAD: global app
app = FastAPI()
pipeline = load_heavy_model()  # 2 minutes, 3 GB VRAM

@app.get("/query")
def query(): ...
```

Problems:
- Importing the module loads the model (breaks testing, IDE indexing, linters)
- No way to inject test configuration
- Can't create multiple apps with different settings

### The Factory Pattern

Instead, we use a **factory function** that creates fresh app instances on demand:

```python
def create_app(settings=None):
    if settings is None:
        settings = Settings()  # read env vars

    app = FastAPI(lifespan=lifespan)
    app.state.app_state = AppState(settings=settings)

    # Register middleware, routes, metrics...
    return app
```

Benefits:
- **Testable**: Tests create apps with custom settings (mock pipeline, fake Redis)
- **Configurable**: Different environments get different settings
- **No side effects on import**: Nothing heavy happens until `create_app()` is called
- **Multiple instances**: Integration tests create isolated apps per test

Uvicorn calls the factory with `--factory`:
```bash
uvicorn src.api.app:create_app --factory --host 0.0.0.0 --port 8000
```

### Lifespan: Startup and Shutdown

The **lifespan** is an async context manager that runs once when the server starts and once when it stops:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # === STARTUP ===
    # 1. Create circuit breaker (instant, in-memory)
    # 2. Connect to Redis + create semantic cache
    # 3. Load RAG pipeline (slow: downloads models, loads index)

    yield  # ← app is running, serving requests

    # === SHUTDOWN ===
    # 1. Disconnect Redis
    # 2. Log shutdown
```

The lifespan pattern replaces the older `@app.on_event("startup")` / `@app.on_event("shutdown")` decorators. It's cleaner because:
- Resources are scoped (created before `yield`, cleaned up after)
- If startup fails, the server doesn't start (fail-fast)
- Works naturally with Python's `with` statement pattern

### AppState: Shared Mutable State

All request handlers need access to the pipeline, cache, and circuit breaker. We store them in an `AppState` dataclass attached to `app.state`:

```python
@dataclass
class AppState:
    pipeline: Optional[RAGPipeline] = None
    cache: Optional[SemanticCache] = None
    circuit_breaker: Optional[CostCircuitBreaker] = None
    settings: Optional[Settings] = None
    start_time: float = field(default_factory=time.time)

    @property
    def is_ready(self) -> bool:
        return self.pipeline is not None

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.start_time
```

Every request handler accesses it via `request.app.state.app_state`. This avoids global variables and makes the app fully testable tests can inject mock pipelines and fake Redis connections.

---

## 4. Configuration Management: Environment Variables

### The Twelve-Factor App

The [Twelve-Factor App](https://12factor.net) methodology says: **store configuration in environment variables**, not in code. This means the same Docker image can run with different models, ports, or cache settings by changing env vars without rebuilding.

### Pydantic BaseSettings

Pydantic's `BaseSettings` class auto-reads from environment variables:

```python
class PipelineConfig(BaseSettings):
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    load_in_4bit: bool = True
    max_new_tokens: int = 80
    temperature: float = 0.1

    model_config = {"env_prefix": "PIPELINE_"}
```

This means:
```bash
# Default: model_name = "Qwen/Qwen2.5-3B-Instruct"
# Override: set env var PIPELINE_MODEL_NAME
export PIPELINE_MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2
```

Each config group has its own prefix to avoid collisions:

| Config Group | Env Prefix | Example Variable |
|-------------|-----------|------------------|
| Pipeline | `PIPELINE_` | `PIPELINE_MODEL_NAME`, `PIPELINE_LOAD_IN_4BIT` |
| Cache | `CACHE_` | `CACHE_REDIS_URL`, `CACHE_SIMILARITY_THRESHOLD` |
| Auth | `AUTH_` | `AUTH_ENABLED`, `AUTH_API_KEY` |
| Circuit Breaker | `CB_` | `CB_COST_LIMIT_EUR`, `CB_WINDOW_SECONDS` |
| API Server | `API_` | `API_LOG_LEVEL`, `API_PORT` |

### Type Coercion

Pydantic automatically converts string env vars to the correct Python type:

```bash
export PIPELINE_LOAD_IN_4BIT=true    # → bool: True
export PIPELINE_MAX_NEW_TOKENS=120   # → int: 120
export PIPELINE_TEMPERATURE=0.3      # → float: 0.3
export CB_COST_LIMIT_EUR=10.0        # → float: 10.0
```

This eliminates bugs from `os.getenv()` + manual parsing.

---

## 5. Request Lifecycle: From HTTP to Answer

### Full Journey of a Query

When a user sends `POST /query {"query": "When did Beyonce become famous?"}`, here is exactly what happens:

```
1. HTTP Request arrives at Uvicorn (port 8000)
   │
2. Uvicorn parses HTTP → ASGI event → passes to FastAPI
   │
3. RequestIDMiddleware runs:
   │  - Reads X-Request-ID from header (or generates uuid4)
   │  - Stores on request.state.request_id
   │
4. FastAPI route matching: POST /query → query_endpoint()
   │
5. Pydantic validation:
   │  - Is "query" present? Is length 3-512? Is top_k 1-20?
   │  - If invalid → 422 error (never reaches pipeline)
   │
6. Auth check (if enabled):
   │  - Read X-API-Key header
   │  - Compare to AUTH_API_KEY env var
   │  - If wrong → 401 Unauthorized
   │
7. Pipeline check:
   │  - Is state.pipeline loaded?
   │  - If not → 503 Service Unavailable
   │
8. Semantic cache check:
   │  - Encode query with all-MiniLM-L6-v2
   │  - Scan Redis for similar queries (cosine > 0.92)
   │  - If HIT → return cached result (skip pipeline!) ← fast path
   │
9. Cache MISS → run pipeline in thread pool:
   │  - Hybrid retrieval (dense + sparse + RRF)
   │  - Cross-encoder reranking
   │  - LLM generation (Qwen 3B, 4-bit, GPU)
   │
10. Build response:
    │  - Extract answer, sources, latency
    │  - Fire-and-forget: cache.set() in background
    │
11. RequestIDMiddleware (response phase):
    │  - Add X-Request-ID to response headers
    │
12. Prometheus metrics update:
    │  - Increment request counter
    │  - Record latency in histogram
    │  - Increment cache hit or miss counter
    │
13. HTTP Response sent back to client
```

### Cache Hit: The Fast Path

When the cache hits, the pipeline doesn't run at all. This is the key performance win:

```
Cache miss: ~3-5 seconds (retrieval + reranking + generation)
Cache hit:  ~5-10 ms (Redis lookup + embedding comparison)
```

That's a 300-1000x speedup for repeated or similar questions.

---

## 6. Async vs Sync: The Executor Bridge

### The Core Problem

FastAPI is **asynchronous** = it uses Python's `asyncio` event loop to handle many requests concurrently. But our RAG pipeline is **synchronous** = it calls `model.generate()` which blocks the CPU/GPU for several seconds.

If we call the synchronous pipeline directly inside an async handler, we **block the entire event loop**. No other requests can be processed until generation finishes:

```python
# BAD: blocks the event loop for 3-5 seconds
@router.post("/query")
async def query(body: QueryRequest):
    result = pipeline.query(body.query)  # BLOCKS!
    return result
```

### How asyncio Works

The event loop is a single thread that processes tasks cooperatively:

```
Event Loop (single thread):
  ┌──────────────────────────────────────────────────────┐
  │ Task 1: read from Redis      (I/O → suspends → resumes when data arrives) │
  │ Task 2: send HTTP response   (I/O → suspends → resumes when sent)         │
  │ Task 3: pipeline.query()     (CPU → BLOCKS ENTIRE LOOP FOR 5 SECONDS!)    │
  └──────────────────────────────────────────────────────┘
```

Tasks 1 and 2 are I/O-bound ==> they `await` and release the loop while waiting for data. Task 3 is CPU/GPU-bound ==> it never `await`s, so it monopolizes the loop.

### The Solution: run_in_executor

`run_in_executor()` offloads a synchronous function to a **thread pool**, freeing the event loop:

```python
@router.post("/query")
async def query(body: QueryRequest):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,           # None = default thread pool
        _run_pipeline,  # sync function
        pipeline,       # argument 1
        body.query,     # argument 2
    )
    return result
```

Now the execution looks like:

```
Event Loop (main thread):              Thread Pool (background thread):
  │                                      │
  │── await run_in_executor() ──────────>│── pipeline.query() (blocking, 5s)
  │                                      │
  │── (free to handle other requests!)   │── ... retrieval ...
  │── Task 2: serve /health → 200        │── ... reranking ...
  │── Task 3: serve /metrics → 200       │── ... generation ...
  │                                      │
  │<── result returned ──────────────────│── done!
  │
  │── build QueryResponse, return
```

The event loop continues serving other requests (health checks, metrics, new queries) while the pipeline runs in the background.

### Why Not Multiple Processes?

You might think: use multiple Uvicorn workers (processes) instead of threads. The problem is GPU memory:

```
1 process = 1 model copy = ~2.5 GB VRAM
2 processes = 2 model copies = ~5.0 GB VRAM
GPU total = 6.4 GB VRAM

2 processes → CUDA out-of-memory error!
```

With `run_in_executor`, we keep a single process (single model in VRAM) but still achieve concurrency for I/O operations.

---

## 7. Streaming: Server-Sent Events (SSE)

### Why Streaming?

When you ask ChatGPT a question, tokens appear one by one. This is **streaming**. Without streaming, the user sees nothing for 3-5 seconds, then the entire answer appears at once. With streaming:

```
Without streaming:                    With streaming:
  [3.2 seconds of nothing]             "Beyonce"         (0.5s)
  "Beyonce became famous in            "became"          (0.6s)
   the late 1990s as the               "famous"          (0.7s)
   lead singer of..."                  "in the"          (0.8s)
                                       "late 1990s..."   (0.9s)
```

Streaming improves **perceived latency** — the user starts reading while the model is still generating.

### What Is SSE?

**Server-Sent Events (SSE)** is a web standard for one-directional streaming from server to client. Unlike WebSockets (bidirectional), SSE is simpler and works over standard HTTP:

```
Client                             Server
  │                                  │
  │── POST /query/stream ──────────>│
  │                                  │── retrieve chunks
  │                                  │── start generation
  │<── event: token                 │
  │    data: {"token": "Beyonce"}    │
  │                                  │
  │<── event: token                 │
  │    data: {"token": " became"}    │
  │                                  │
  │<── event: token                 │
  │    data: {"token": " famous"}    │
  │                                  │
  │<── event: done                  │
  │    data: {"request_id": "abc"}   │
  │                                  │
  │── connection closed ────────────│
```

Each **event** has a type (`token` or `done`) and a data payload (JSON). The client reads events as they arrive — no polling, no buffering.

### How Streaming Works Internally

Streaming requires coordinating three things:
1. A **background thread** running `model.generate()` (GPU-bound, blocking)
2. A **TextIteratorStreamer** that intercepts generated tokens
3. An **async generator** that yields SSE events

```
Thread Pool                     Event Loop
  │                               │
  │ model.generate(               │
  │   streamer=TextIterator)      │
  │                               │
  │ → token "Beyonce"             │
  │   → streamer queue ──────────>│── yield SSE event: "Beyonce"
  │                               │
  │ → token " became"             │
  │   → streamer queue ──────────>│── yield SSE event: " became"
  │                               │
  │ → [generation done]           │
  │   → streamer signals end ────>│── yield SSE event: "done"
  │                               │── join thread
```

The `TextIteratorStreamer` from HuggingFace acts as a bridge — it has a thread-safe queue. The generation thread puts tokens into the queue, and the async event loop reads them out.

```python
# Simplified version of our streaming endpoint
async def event_generator():
    # 1. Retrieve context (in executor)
    retrieval_result = await loop.run_in_executor(None, _run_pipeline, ...)

    # 2. Build prompt from retrieved chunks
    prompt = generator.build_prompt(query, chunks)

    # 3. Create streamer (thread-safe queue)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)

    # 4. Start generation in background thread
    thread = Thread(target=lambda: generator._generate_text(prompt, streamer=streamer))
    thread.start()

    # 5. Yield tokens as they arrive
    for token in streamer:      # blocks until next token is ready
        yield {"event": "token", "data": json.dumps({"token": token})}
        await asyncio.sleep(0)  # yield control to event loop

    yield {"event": "done", "data": json.dumps({"request_id": request_id})}
    thread.join()
```

The `await asyncio.sleep(0)` on line is critical — without it, the for loop would monopolize the event loop while waiting for the next token.

---

## 8. Semantic Caching with Redis

### Traditional Caching vs Semantic Caching

**Traditional cache**: exact string match. "What is the capital of France?" and "What's the capital of France?" are different keys → cache miss.

**Semantic cache**: meaning-based match. Both questions have cosine similarity > 0.92 → cache hit. The answer is the same, so why recompute it?

```
Traditional Cache:
  "What is the capital of France?"  →  HIT  ✓
  "What's the capital of France?"   →  MISS ✗  (different string!)
  "capital of France"               →  MISS ✗
  "What is France's capital?"       →  MISS ✗

Semantic Cache (cosine threshold = 0.92):
  "What is the capital of France?"  →  HIT  ✓
  "What's the capital of France?"   →  HIT  ✓  (similarity = 0.97)
  "capital of France"               →  HIT  ✓  (similarity = 0.93)
  "What is France's capital?"       →  HIT  ✓  (similarity = 0.95)
  "Who is the president of France?" →  MISS ✗  (similarity = 0.71)
```

### How It Works

#### Step 1: Encoding Queries

Each query is embedded using `all-MiniLM-L6-v2`, a small (22M parameter) sentence embedding model that runs on CPU:

```
"What is the capital of France?"
    → [0.23, -0.15, 0.67, ..., 0.42]   (384 dimensions)
    → L2-normalized: ||v|| = 1.0
```

L2 normalization ensures that cosine similarity equals the dot product:

```
cosine_similarity(a, b) = (a · b) / (||a|| × ||b||)

If ||a|| = 1 and ||b|| = 1:
cosine_similarity(a, b) = a · b    ← simple dot product!
```

#### Step 2: Cache Lookup (get)

When a new query arrives:
1. Encode it → `query_embedding`
2. Scan all cached entries in Redis
3. For each cached entry, compute `dot(query_embedding, cached_embedding)`
4. Find the best match
5. If best similarity >= 0.92 → return cached result

```python
async def get(self, query: str):
    query_embedding = self._encode(query)       # step 1
    best_match = None
    best_similarity = -1.0

    async for key in self._redis.scan_iter(match="rag:cache:*"):  # step 2
        entry = json.loads(await self._redis.get(key))
        cached_embedding = np.array(entry["embedding"])
        similarity = float(np.dot(query_embedding, cached_embedding))  # step 3

        if similarity > best_similarity:   # step 4
            best_similarity = similarity
            best_match = entry

    if best_similarity >= self.similarity_threshold:  # step 5
        return best_match["result"]
    return None
```

#### Step 3: Cache Store (set)

After a cache miss, the query result is stored:

```python
async def set(self, query: str, result: dict):
    embedding = self._encode(query)
    key = f"rag:cache:{uuid4().hex}"

    entry = {
        "embedding": embedding.tolist(),    # for similarity comparison
        "query": query,                      # for debugging
        "result": result,                    # the actual cached response
        "created_at": time.time(),
    }

    await self._redis.set(key, json.dumps(entry), ex=self.ttl_seconds)
```

The `ex=self.ttl_seconds` sets an **expiration time** (default: 86400 seconds = 24 hours). After that, Redis automatically deletes the entry.

### Why Redis?

Redis is an in-memory key-value store. It's used here because:

1. **Fast**: All data in RAM → microsecond reads
2. **Persistent**: Can survive restarts (optional)
3. **TTL support**: Entries auto-expire (no manual cleanup)
4. **LRU eviction**: When memory is full, the least recently used entry is evicted
5. **Separate process**: Cache survives API restarts
6. **Industry standard**: Battle-tested, excellent tooling

Our Redis instance is configured with `--maxmemory 256mb --maxmemory-policy allkeys-lru`. This means: use at most 256 MB of RAM, and when full, evict the oldest unused entry.

### The O(N) Scan Trade-off

Our implementation scans **all** cache entries for every lookup. This is O(N) where N = number of cached entries. For a production system with millions of entries, this would be too slow.

For our use case (~100-1000 entries), the scan takes <10ms which is acceptable. For larger scale, you would use:
- **Approximate Nearest Neighbor** (ANN) search: libraries like FAISS or Annoy
- **Redis Vector Search**: Redis Stack has native vector similarity search
- **Dedicated vector databases**: Pinecone, Weaviate, Qdrant

### The Threshold Choice: 0.92

Why 0.92 and not 0.8 or 0.99?

```
Threshold = 0.99  → Almost exact matches only. Very few hits. Safe but wasteful.
Threshold = 0.92  → Paraphrases and minor variations. Good balance.
Threshold = 0.80  → Loosely related questions. Dangerous — might return wrong answers!
```

```
"What is the capital of France?" vs "What's France's capital?"
  → similarity = 0.95 → HIT at 0.92 ✓ (correct, same question)

"What is the capital of France?" vs "What is the capital of Germany?"
  → similarity = 0.88 → MISS at 0.92 ✓ (correct, different question!)

"What is the capital of France?" vs "Tell me about France"
  → similarity = 0.72 → MISS at 0.92 ✓ (correct, different intent)
```

If we used 0.80, the France/Germany pair might match → returning "Paris" for a question about Germany. The 0.92 threshold is conservative enough to avoid this.

### Fire-and-Forget Cache Set

After computing a response, we store it in cache using a **fire-and-forget** pattern:

```python
asyncio.create_task(cache.set(body.query, response_dict))
```

`create_task()` schedules the cache write in the background. The response is sent immediately — the user doesn't wait for Redis write confirmation. If the write fails, we just skip caching (no error to the user).

---

## 9. Circuit Breaker Pattern

### The Problem: Runaway Costs

If you replace our local Qwen model with a cloud API (OpenAI, Anthropic), every query costs money:

```
1 query   ≈ $0.002 (GPT-4o-mini)
100 queries = $0.20
10,000 queries = $20.00
DDoS attack / bug in a loop = $$$$ before you notice
```

A **circuit breaker** automatically stops requests when spending exceeds a budget.

### How a Circuit Breaker Works

The name comes from electrical engineering. A circuit breaker in your house trips when too much current flows, preventing a fire. Our software circuit breaker trips when too much money is spent, preventing a surprise bill.

```
                ┌─────────────────────┐
                │                     │
                │   CLOSED (normal)   │── cost < limit → allow request
                │                     │
                └──────────┬──────────┘
                           │ cost >= limit
                           ▼
                ┌─────────────────────┐
                │                     │
                │   OPEN (blocked)    │── reject all requests with 429
                │                     │
                └──────────┬──────────┘
                           │ window expires
                           ▼
                   (reset to CLOSED)
```

Two states:
- **CLOSED**: Normal operation. Requests go through. Each request's cost is recorded.
- **OPEN**: Budget exceeded. All requests are rejected with "429 Too Many Requests".

### Implementation Details

```python
class CostCircuitBreaker:
    def __init__(self, cost_limit_eur=5.0, window_seconds=3600):
        self.cost_limit_eur = cost_limit_eur
        self.window_seconds = window_seconds
        self._lock = threading.Lock()        # thread-safe
        self._cumulative_cost_eur = 0.0
        self._window_start = time.time()

    @property
    def is_open(self) -> bool:
        with self._lock:
            self._maybe_reset_window()  # auto-reset if window expired
            return self._cumulative_cost_eur >= self.cost_limit_eur

    def record_spend(self, input_tokens, output_tokens) -> float:
        cost = (input_tokens / 1000) * self.cost_per_1k_input \
             + (output_tokens / 1000) * self.cost_per_1k_output
        with self._lock:
            self._maybe_reset_window()
            self._cumulative_cost_eur += cost
        return cost
```

Key design decisions:

1. **Thread-safe** (`threading.Lock`): The executor runs pipeline.query() in a thread pool. Multiple threads might call `record_spend()` simultaneously. Without the lock, `_cumulative_cost_eur` could be corrupted by a race condition.

2. **Auto-reset window**: If 1 hour has passed since the window started, the cost counter resets to 0. This prevents permanent lockout — the breaker opens for the current hour, then closes for the next hour.

3. **EUR, not USD**: A conscious decision — adapt the currency to your billing context.

### Cost Estimation

Even though our local model is free, we track **hypothetical cost** as if we were using a cloud API. This is useful for:
- Estimating what it would cost to move to GPT-4o
- Setting realistic budgets before migrating to cloud
- Monitoring usage patterns even with free models

---

## 10. Authentication & Middleware

### API Key Authentication

Our API uses a simple API key scheme:

```
Client                              Server
  │                                   │
  │── POST /query                     │
  │   Header: X-API-Key: my-secret    │
  │   Body: {"query": "..."}          │
  │                                   │── check AUTH_ENABLED
  │                                   │── compare X-API-Key to AUTH_API_KEY
  │                                   │── match → proceed to pipeline
  │<── 200 OK
```

If the key is wrong or missing:
```
  │── POST /query                     │
  │   (no X-API-Key header)           │
  │                                   │── AUTH_ENABLED=true
  │                                   │── no key → reject
  │<── 401 Unauthorized               │
  │   {"detail": "Invalid or missing API key"} │
```

The auth is a **FastAPI dependency** = a function that runs before the route handler:

```python
async def verify_api_key(request: Request, key: str = Security(api_key_header)):
    settings = request.app.state.app_state.settings
    if not settings.auth.enabled:
        return None  # Auth disabled → allow everything
    if not key or key != settings.auth.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return key
```

Applied to routes via `dependencies=[Depends(verify_api_key)]`:
- `/query`, `/query/stream`, `/batch` → require auth (if enabled)
- `/health`, `/health/ready`, `/health/gpu`, `/health/stats` → no auth (monitoring always works)

### Why Disable by Default?

Auth is disabled by default (`AUTH_ENABLED=false`) for developer convenience. During local development, you don't want to type an API key for every `curl` test. In production, set `AUTH_ENABLED=true` and `AUTH_API_KEY=<random-secret>` in `.env`.

### Middleware: Request ID Tracing

**Middleware** is code that runs on every request, before and after the route handler. Our `RequestIDMiddleware` assigns a unique ID to each request:

```python
class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # Before route handler:
        request_id = request.headers.get("X-Request-ID", uuid4().hex)
        request.state.request_id = request_id

        # Run the route handler:
        response = await call_next(request)

        # After route handler:
        response.headers["X-Request-ID"] = request_id
        return response
```

This is critical for **distributed tracing**. When a user reports "my query was slow", you need to find the corresponding log entry among millions. The request ID links everything together:

```
# Request
POST /query
X-Request-ID: abc-123

# Response header
X-Request-ID: abc-123

# Response body
{"answer": "...", "request_id": "abc-123"}

# Server logs
[abc-123] Cache miss for query "..."
[abc-123] Pipeline completed in 3200ms
[abc-123] Response sent
```

If the client provides their own `X-Request-ID`, we use it (preserves end-to-end tracing across microservices). If not, we generate one (uuid4 = 128-bit random ID, collision probability ≈ 0).

---

## 11. Observability: Metrics, Health Checks, Logging

### Why Observability?

A production system is a black box. Users send requests, responses come back. But:
- Is it getting slower over time?
- Is the cache actually helping?
- Is the GPU running out of VRAM?
- Are errors increasing?
- How close are we to the cost limit?

**Observability** makes the black box transparent. It has three pillars:
1. **Metrics** (numbers over time): latency, request count, cache hit rate
2. **Logs** (structured events): per-request details, errors, warnings
3. **Traces** (request journey): which middleware ran, how long each step took

### Prometheus Metrics

**Prometheus** is a time-series database that periodically scrapes (pulls) metrics from your application. Our API exposes a `/metrics` endpoint in Prometheus format:

```
# HELP rag_requests_total Total HTTP requests
# TYPE rag_requests_total counter
rag_requests_total{method="POST",endpoint="/query",status="200"} 42

# HELP rag_query_latency_seconds End-to-end query latency
# TYPE rag_query_latency_seconds histogram
rag_query_latency_seconds_bucket{le="1.0"} 5
rag_query_latency_seconds_bucket{le="2.0"} 15
rag_query_latency_seconds_bucket{le="3.0"} 35
rag_query_latency_seconds_bucket{le="5.0"} 40
rag_query_latency_seconds_bucket{le="+Inf"} 42

# HELP rag_cache_hits_total Semantic cache hits
# TYPE rag_cache_hits_total counter
rag_cache_hits_total 28
```

#### Metric Types

| Type | What It Is | Example |
|------|-----------|---------|
| **Counter** | Only goes up. Resets to 0 on restart. | `rag_requests_total`, `rag_cache_hits_total`, `rag_errors_total` |
| **Histogram** | Counts values in configurable buckets. | `rag_query_latency_seconds` (buckets: 0.5s, 1s, 2s, 3s, 5s, ...) |
| **Gauge** | Goes up and down. Current value. | `rag_gpu_vram_used_gb`, `rag_cache_size`, `rag_circuit_breaker_cumulative_cost_eur` |

#### Why Histograms, Not Averages?

Average latency is misleading. If 99 requests take 1s and 1 request takes 60s, the average is 1.6s — hiding a terrible outlier. Histograms let you compute **percentiles**:

```
p50 = median (half are faster, half slower)
p95 = 95th percentile (5% of users experience worse)
p99 = 99th percentile (1% of users experience worse)

Example from our dashboard:
  p50 = 2.1s   (typical experience)
  p95 = 4.8s   (occasionally slow)
  p99 = 8.2s   (rare but painful)
```

Our histogram buckets are calibrated from Step 7 evaluation data:
```python
QUERY_LATENCY = Histogram(
    "rag_query_latency_seconds",
    "End-to-end query latency",
    buckets=[0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0],
)
```

### Health Check Endpoints

Health checks let orchestrators (Kubernetes, Docker, load balancers) know if the service is functional:

| Endpoint | Purpose | When It Returns "not ok" |
|----------|---------|--------------------------|
| `GET /health` | **Liveness probe**: Is the process alive? | Never (if it responds at all, it's alive) |
| `GET /health/ready` | **Readiness probe**: Can it serve queries? | Pipeline not loaded yet (model downloading) |
| `GET /health/gpu` | **GPU check**: Is the model in VRAM? | CUDA unavailable, model not on GPU |
| `GET /health/stats` | **Diagnostics**: Cache, breaker, pipeline stats | N/A (always returns data) |

#### Liveness vs Readiness

This distinction matters for orchestrators:

```
Timeline:
  0s     Container starts
  0-120s Model loading (download weights, quantize, move to GPU)
  120s+  Ready to serve queries

Liveness probe:  "Is the process running?" → YES from 0s
Readiness probe: "Can it serve queries?"   → NO until 120s, then YES

If readiness = NO: orchestrator stops sending traffic (but doesn't restart)
If liveness = NO:  orchestrator kills and restarts the container
```

This prevents users from getting 503 errors during model loading while also allowing the orchestrator to detect a truly crashed process.

### The Grafana Dashboard

**Grafana** is a visualization tool that reads from Prometheus and displays dashboards. Our pre-built dashboard has 8 panels:

```
┌───────────────────────────┬────────────────────────────┐
│  Request Rate (req/s)     │  Query Latency(p50/p95/p99)│
│  rag_requests_total       │  rag_query_latency_seconds │
├───────────┬───────────────┼──────────────┬─────────────┤
│ Cache Hit │ Cache Hits vs │ GPU VRAM (GB)│ Circuit Cost│
│ Rate (%)  │ Misses        │ (gauge)      │ (EUR)       │
├───────────┴───────────────┼──────────────┴─────────────┤
│  Error Rate               │  Retrieval vs Generation   │
│  rag_errors_total         │  Latency (p95 comparison)  │
└───────────────────────────┴────────────────────────────┘
```

The dashboard is **provisioned** — it loads automatically when Grafana starts. No manual setup needed. This is achieved through configuration files mounted into the Grafana container:

```
configs/grafana/
  datasources.yml           → tells Grafana where Prometheus is
  dashboards/
    provider.yml            → tells Grafana to load dashboards from /var/lib/grafana/dashboards
    rag_dashboard.json      → the actual dashboard definition (panels, queries, layout)
```

### The Scrape Loop

Prometheus pulls metrics from our API every 15 seconds:

```
Prometheus                          RAG API
  │                                   │
  │── GET /metrics ──────────────────>│
  │                                   │── collect all counters/histograms/gauges
  │<── 200 OK                         │── return text/plain (Prometheus format)
  │                                   │
  │── [store time series]             │
  │── [15 seconds later, repeat]      │
```

This **pull model** is a key Prometheus design choice:
- No push configuration needed on the API side
- Prometheus knows when a target is down (scrape fails)
- Single source of truth for scrape intervals
- API just exposes a `/metrics` endpoint — stateless

---

## 12. Containerization: Docker & Docker Compose

### What Is Docker?

Docker is a tool for packaging and running applications in isolated environments called **containers**. A container is like a lightweight virtual machine = it has its own filesystem, network, and processes, but shares the host's OS kernel (much faster than a VM).

### Why Docker?

Without Docker:
```
Developer A: "It works on my machine" (Python 3.11, Ubuntu 22.04, CUDA 12.1)
Developer B: "It crashes on mine" (Python 3.9, macOS, no CUDA)
Ops team:    "It doesn't start in production" (Python 3.12, Debian, CUDA 11.8)
```

With Docker:
```
Same image → same behavior everywhere.
Docker image = your code + all dependencies + exact Python version + system libs
```

### The Dockerfile: Multi-Stage Build

Our Dockerfile uses a **multi-stage build** to keep the final image small:

```dockerfile
# Stage 1: BUILDER - install dependencies (big, temporary)
FROM python:3.11-slim AS builder
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: RUNTIME - copy only what's needed (small, final)
FROM python:3.11-slim AS runtime
COPY --from=builder /install /usr/local   # copy installed packages
COPY src/ src/                             # copy application code
COPY data/ data/                           # copy FAISS index
```

Why multi-stage?
- Stage 1 installs build tools (gcc, git) needed to compile packages
- Stage 2 doesn't have build tools, only runtime dependencies
- Final image is ~1 GB smaller because compiler toolchains are excluded

### Dockerfile Walkthrough

```dockerfile
# Base image: Python 3.11 (slim = no unnecessary system packages)
FROM python:3.11-slim AS builder
WORKDIR /build

# Install build dependencies (needed for compiling C extensions)
RUN apt-get update && apt-get install -y build-essential git

# Install Python packages to a separate prefix (for copying later)
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt
```

```dockerfile
FROM python:3.11-slim AS runtime
WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Security: run as non-root user
RUN useradd --create-home appuser

# Copy application code and data
COPY src/ src/
COPY data/ data/
COPY index/ index/

# Install package for "src" imports to work
RUN pip install -e .

USER appuser   # switch to non-root (security best practice)

# Default environment variables (overridable at runtime)
ENV PIPELINE_MODEL_NAME=Qwen/Qwen2.5-3B-Instruct \
    PIPELINE_LOAD_IN_4BIT=true \
    CACHE_REDIS_URL=redis://redis:6379/0
```

```dockerfile
EXPOSE 8000  # documentation: this container listens on port 8000

# Health check: Docker periodically runs this command
# start-period=120s: don't start checking until model is loaded
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/ready')"

# The command that runs when the container starts
# --workers 1: single worker because GPU can only hold one model copy
ENTRYPOINT ["uvicorn", "src.api.app:create_app", "--factory", \
            "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

### Docker Compose: Orchestrating Multiple Containers

Docker Compose manages multiple containers as a single stack. Our stack has 4 services:

```yaml
services:
  rag-api:    # Our FastAPI application (GPU)
  redis:      # Semantic cache storage
  prometheus: # Metric collection
  grafana:    # Dashboard visualization
```

#### Service Dependencies

```
grafana → reads from → prometheus → scrapes → rag-api → uses → redis
```

Docker Compose handles startup order via `depends_on`:

```yaml
rag-api:
  depends_on:
    redis:
      condition: service_healthy  # wait for Redis health check to pass
```

This ensures Redis is up before the API tries to connect.

#### GPU Passthrough

Docker containers don't have GPU access by default. We request it explicitly:

```yaml
rag-api:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

This tells Docker to pass through one NVIDIA GPU. Requires:
- NVIDIA Container Toolkit installed on the host
- Docker Desktop with WSL2 backend (on Windows)

#### Memory Limits

Monitoring containers are capped to protect GPU memory:

```yaml
redis:
  deploy:
    resources:
      limits:
        memory: 256M    # enough for ~10K cache entries

prometheus:
  deploy:
    resources:
      limits:
        memory: 512M

grafana:
  deploy:
    resources:
      limits:
        memory: 512M
```

Total overhead: 256 + 512 + 512 = 1.3 GB for infrastructure. The remaining ~15.7 GB RAM is available for the API process and model loading.

#### Named Volumes

Data persists across container restarts via named volumes:

```yaml
volumes:
  model-cache:      # HuggingFace model weights (~3 GB)
  redis-data:       # Cache entries
  prometheus-data:  # Metric history
  grafana-data:     # Dashboard settings
```

Without volumes, restarting the API container would re-download the 3 GB model every time.

### Running the Stack

```bash
# Build and start all 4 containers
docker compose up -d

# Check status
docker compose ps

# View API logs
docker compose logs rag-api

# Stop everything
docker compose down

# Stop and delete all data (volumes)
docker compose down -v
```

---

## 13. GPU Constraints & Single-Worker Design

### The VRAM Budget

Our RTX 3060 Laptop GPU has 6.4 GB VRAM. Here's how it's used:

```
Component                      VRAM Usage
─────────────────────────────────────────
Qwen2.5-3B (4-bit quantized)   ~2.2 GB
BGE-large-en-v1.5 (embeddings) ~1.3 GB
Cross-encoder reranker          ~0.4 GB
CUDA overhead + KV cache        ~1.0 GB
─────────────────────────────────────────
Total                           ~4.9 GB
Available headroom              ~1.5 GB
```

This leaves ~1.5 GB of headroom. Not enough for a second model copy.

### Why Not 4-bit + 8-bit Mixed?

One might think: load the embedding model in 8-bit to save even more VRAM. But:
- Embedding models are already small (~1.3 GB in fp16)
- 8-bit quantization can degrade embedding quality (important for retrieval accuracy)
- The bottleneck is the LLM (3B parameters), not the embedding model

### Why Single Worker?

```
Workers = 1: 1 × 4.9 GB = 4.9 GB → fits in 6.4 GB ✓
Workers = 2: 2 × 4.9 GB = 9.8 GB → OOM on 6.4 GB ✗
```

Each Uvicorn worker is a separate OS process with its own copy of the model in VRAM. There is no GPU memory sharing between processes.

To handle multiple concurrent users with a single worker, we use:
1. `run_in_executor()` for async request handling (see Section 6)
2. Semantic caching for repeated queries (see Section 8)
3. SSE streaming for perceived responsiveness (see Section 7)

### Batch Processing: Sequential, Not Parallel

Our `/batch` endpoint processes queries **sequentially**, not in parallel:

```python
for query_text in body.queries:
    result = await loop.run_in_executor(None, _run_pipeline, pipeline, query_text)
    results.append(result)
```

Why not parallel? Because `model.generate()` uses the GPU. Two concurrent GPU operations on a single GPU don't run in parallel they either interleave (slower due to context switching) or one waits for the other (no benefit). Sequential processing is simpler and equally fast.

---

## 14. Key Takeaways

1. **Deployment is not optional**: A model that only works via Python import is not a product. Wrapping it in an HTTP API (FastAPI + Uvicorn) makes it usable by anyone.

2. **The factory pattern enables testing**: `create_app(settings)` lets you inject mock pipelines and fake Redis, making 82 tests run without a GPU.

3. **Async + executor bridges two worlds**: FastAPI's event loop handles concurrent I/O (Redis, HTTP) while `run_in_executor` offloads blocking GPU work to a thread pool.

4. **Semantic caching is a multiplier**: A cache hit (5ms) is 300-1000x faster than a pipeline run (3-5s). Even a 30% hit rate dramatically improves average latency.

5. **The circuit breaker is a safety net**: Runaway cost is a real risk with cloud LLMs. A cost-based circuit breaker with automatic window reset prevents bill shock without permanent lockout.

6. **Observability is not a luxury**: Without Prometheus metrics and Grafana dashboards, you're blind to degradation, cache effectiveness, and GPU health. The `/health` endpoints let orchestrators make informed decisions.

7. **Docker ensures reproducibility**: "Works on my machine" is solved by containerizing everything — exact Python version, exact package versions, exact system libraries.

8. **GPU VRAM is the bottleneck**: On a 6 GB GPU, everything follows from the constraint: single worker, sequential batch, CPU-only cache model. Design with your hardware limits in mind, not theoretical ideals.

9. **Configuration via environment variables**: The same Docker image serves development, staging, and production with different env vars. No code changes, no rebuilds.

10. **Streaming improves UX, not throughput**: SSE streaming doesn't make generation faster — it makes the user *feel* like it's faster by showing tokens as they're generated.

---

## References

- FastAPI Documentation: https://fastapi.tiangolo.com
- Uvicorn ASGI Server: https://www.uvicorn.org
- Prometheus Monitoring: https://prometheus.io/docs/introduction/overview/
- Grafana Dashboards: https://grafana.com/docs/grafana/latest/
- Docker Documentation: https://docs.docker.com
- The Twelve-Factor App: https://12factor.net
- Redis Documentation: https://redis.io/documentation
- Nygard, M. (2007). "Release It! Design and Deploy Production-Ready Software"
- Fowler, M. (2014). "CircuitBreaker" — https://martinfowler.com/bliki/CircuitBreaker.html
- HuggingFace TextIteratorStreamer: https://huggingface.co/docs/transformers/main_classes/text_generation

---

Continue to **[experiments/step8_deployment_results.md](../experiments/step8_deployment_results.md)** for deployment benchmarks →
