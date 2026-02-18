# xordb

```
  ██╗  ██╗ ██████╗ ██████╗ ██████╗ ██████╗ 
  ╚██╗██╔╝██╔═══██╗██╔══██╗██╔══██╗██╔══██╗
   ╚███╔╝ ██║   ██║██████╔╝██║  ██║██████╔╝
   ██╔██╗ ██║   ██║██╔══██╗██║  ██║██╔══██╗
  ██╔╝ ██╗╚██████╔╝██║  ██║██████╔╝██████╔╝
  ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═════╝ ╚═════╝ 
```

**The fastest binary semantic cache for LLM apps.**  
*Offline. Private. Zero cloud costs.*

Built on Hyperdimensional Computing. Written in Go.

License: GPL-3.0 (see [LICENSE](LICENSE)).

---

## Why I built this

While building LLM agents, I kept running into the same inference cost problem.
Every user query — no matter how repetitive — fires a full round-trip to the
model. You're paying for the same answer over and over.

The obvious fix is caching. But normal caches match on exact strings, and users
never ask the same question the same way twice:

```
"What is the capital of India?"
"capital city of india"
"India's capital?"
```

Three different strings. Same intent. A hash-based cache misses all of them.

Even the "smart" workaround of structuring prompts with a static system message
first and the dynamic user query last — hoping the LLM provider's KV-cache
gives you a prefix hit — barely helps. The user portion still varies, the cache
miss rate stays high, and you're still paying per token.

I needed a cache that understands *meaning*, not strings. One that runs locally,
adds zero latency from network calls, and doesn't send my users' queries to yet
another third-party API.

So I built xordb.

---

## How it works

xordb encodes text into a **hypervector** — a 10,000-bit binary array where
meaning is distributed across every bit. Similarity between two hypervectors is
measured with Hamming distance: a bitwise XNOR + popcount that runs in
nanoseconds.

Three primitives power everything:

| Operation | Purpose | Implementation |
|-----------|---------|----------------|
| `Bundle`  | Combine concepts ("A or B") | majority vote, bit-by-bit |
| `Bind`    | Associate concepts ("A means B") | XOR (self-inverse) |
| `Similarity` | How alike are two things? | popcount / Hamming |

xordb ships with two encoding modes:

| Mode | How it works | Quality | Dependencies |
|------|-------------|---------|-------------|
| **Built-in** (n-gram HDC) | Character n-grams with positional binding | Good for surface-level similarity | Zero |
| **MiniLM** (`xordb/embed`) | Local ONNX transformer model → binary projection | Near-OpenAI semantic quality | `onnxruntime_go` + model file |

Pick the one that fits your use case. The core library stays zero-dependency
either way.

---

## Real-world scenarios

### LLM response caching

The primary use case. Your agent answers "what is the capital of India?" once,
and every rephrase — "India's capital city?", "capital of india", "tell me
india capital" — hits the cache instead of burning another API call.

```go
db.Set("what is the capital of india", llmResponse)

v, ok, _ := db.Get("capital city of india")
// ok=true, v=llmResponse — saved one LLM call
```

At 1000 queries/day with 60% semantic overlap, that's 600 fewer inference calls.
At $0.01/call, ~$180/month saved from a single cache.

### Multi-turn agent deduplication

Agents loop. A ReAct agent might re-derive the same tool call three times in one
conversation. A planning agent might re-ask the same sub-question across
different branches. xordb catches these before they hit the model.

```go
// Agent asks a sub-question during planning
db.Set("what are the business hours of Acme Corp", toolResult)

// Later in the same conversation, different phrasing
v, ok, _ := db.Get("Acme Corp business hours")
// ok=true — no redundant tool call
```

### RAG retrieval caching

Retrieval-Augmented Generation pipelines run an embedding + vector search on
every query. If your users ask similar questions, you're re-retrieving the same
documents. Cache the retrieval results:

```go
db.Set("how do I reset my password", retrievedChunks)

db.Get("password reset instructions")
// Cache hit — skip the entire retrieval pipeline
```

### Prompt template deduplication

When you're generating prompts from templates, slight variations in user input
produce "different" prompts that are semantically identical:

```
"Summarize this article: [article about climate change]"
"Please summarize the following article: [same article about climate change]"
```

xordb catches these because the meaning hasn't changed, even though the wrapper
text did.

### Chatbot FAQ matching

Pre-populate the cache with your FAQ pairs. User questions get matched
semantically without needing a full NLP pipeline:

```go
db.Set("what is your refund policy", "30-day money-back guarantee")
db.Set("how do I contact support", "Email support@example.com")

db.Get("can I get a refund")       // hits refund policy
db.Get("how to reach your team")   // hits contact support
```

### Edge / offline inference

xordb runs entirely in-process. No Redis, no server, no network. This makes it
ideal for:

- **Mobile apps** — cache LLM responses on-device
- **IoT / embedded** — edge devices with intermittent connectivity
- **Privacy-sensitive apps** — user queries never leave the process
- **Air-gapped environments** — no external dependencies at runtime

---

## Quick start

### Mode 1: Zero dependencies (built-in n-gram encoder)

Good for fuzzy string matching, typo tolerance, and surface-level similarity.
No setup required.

```go
import "xordb"

db := xordb.New(xordb.WithThreshold(0.65))

db.Set("what is the capital of india", "Delhi")

v, ok, sim := db.Get("capital city of india")
// ok=true, sim≈0.72, v="Delhi"

_, ok, _ = db.Get("how do you bake a chocolate cake")
// ok=false — unrelated, correctly rejected
```

### Mode 2: Semantic quality (MiniLM encoder)

Real semantic understanding. "author of ramayana" matches "who wrote ramayana"
even though the words barely overlap. Runs a quantized MiniLM-L6-v2 model
locally via ONNX Runtime — no API calls, no cloud, no per-query cost.

```bash
# One-time: download the model (~90MB)
go run xordb/embed/cmd/xordb-model download
```

```go
import (
    "xordb"
    "xordb/embed"
)

enc, err := embed.NewMiniLMEncoder()
if err != nil {
    log.Fatal(err)
}
defer enc.Close()

db := xordb.NewWithEncoder(enc,
    xordb.WithThreshold(0.75),
    xordb.WithCapacity(10000),
)

db.Set("what is the capital of india", "Delhi")
db.Set("who wrote ramayana", "Valmiki")

v, ok, sim := db.Get("india's capital city")
// ok=true, sim≈0.85+, v="Delhi"

v, ok, sim = db.Get("author of ramayana")
// ok=true — real semantic match, not just string overlap
```

### How to choose

| | Built-in (n-gram) | MiniLM (`xordb/embed`) |
|---|---|---|
| **Dependencies** | Zero | `onnxruntime_go` + model file |
| **Semantic quality** | Surface-level (word overlap) | Near-OpenAI (MTEB benchmarks) |
| **"capital of india" ↔ "india's capital"** | ✅ hits (shared n-grams) | ✅ hits (semantic match) |
| **"who wrote ramayana" ↔ "author of ramayana"** | ❌ misses (no word overlap) | ✅ hits (semantic match) |
| **Encode latency** | ~500µs | ~5ms |
| **Binary size** | ~2MB | ~2MB + 90MB model |
| **Best for** | Typo tolerance, fuzzy matching, edge devices | LLM caching, RAG dedup, chatbot FAQ |

---

## API

### Creating a DB

**With built-in encoder:**

```go
db := xordb.New(opts ...Option)
```

| Option | Default | Description |
|--------|---------|-------------|
| `WithDims(n)` | `10000` | Hypervector dimension. Higher = more accurate, more memory. |
| `WithThreshold(t)` | `0.82` | Minimum similarity for a cache hit. Range: `(0, 1]`. |
| `WithCapacity(n)` | `1024` | Max entries. Oldest evicted when exceeded (LRU). |
| `WithNGramSize(n)` | `3` | Character n-gram window. |
| `WithSeed(s)` | `0` | Encoder seed. DBs with different seeds are incompatible. |
| `WithStripPunctuation(v)` | `false` | Strip punctuation before encoding. |

**With custom encoder (e.g. MiniLM):**

```go
db := xordb.NewWithEncoder(enc, opts ...Option)
```

Accepts any `hdc.Encoder` implementation. Only `WithThreshold` and `WithCapacity`
are used — encoding options are controlled by the encoder itself.

### MiniLM encoder options

```go
enc, err := embed.NewMiniLMEncoder(
    embed.WithModelPath("/path/to/model.onnx"),  // default: auto-detect
    embed.WithMaxSeqLen(128),                      // default: 128
    embed.WithBinaryDims(10000),                   // default: 10000
    embed.WithProjectionSeed(0xDBCAFE),            // default: deterministic
)
```

### Methods

```go
db.Set(key string, value any)
```
Store any value under a string key. If the exact key exists, update it and
promote to most-recently-used.

```go
db.Get(key string) (value any, hit bool, similarity float64)
```
Return the value under the most similar key at or above the threshold.
Returns `(nil, false, 0)` on a miss.

```go
db.Delete(key string) bool
```
Remove the entry with the **exact** key string. Returns true if found.

```go
db.Len() int
```
Current number of cached entries.

```go
db.Stats() xordb.Stats
```

```go
type Stats struct {
    Entries     int
    Hits        uint64
    Misses      uint64
    Sets        uint64
    HitRate     float64
    AvgSimOnHit float64
}
```

---

## Model management

The MiniLM encoder needs a one-time model download (~90MB). The `xordb-model`
CLI handles this:

```bash
# Download the model
go run xordb/embed/cmd/xordb-model download

# Check status
go run xordb/embed/cmd/xordb-model info

# Print model path
go run xordb/embed/cmd/xordb-model path
```

The model is stored at `~/.local/share/xordb/models/all-MiniLM-L6-v2.onnx`
(or `$XDG_DATA_HOME/xordb/models/`). Override with `XORDB_MODEL_PATH`.

---

## Performance

### HDC primitives (`hdc/`)

| Operation | Time | Notes |
|-----------|------|-------|
| `Similarity` | **67 ns** | Pure POPCNT, no alloc |
| `Bind` | 365 ns | XOR + allocation |
| `Permute` | 520 ns | Clone + cyclic shift |
| `Bundle(10)` | 141 µs | Majority vote, 10 vecs × 10k bits |

### Cache lookups

| Operation | Time | Notes |
|-----------|------|-------|
| `Set` (n-gram) | 532 µs | Encode + insert |
| `Get` — 100 entries | 460 µs | Encode + scan |
| `Get` — 1,000 entries | 820 µs | Encode + scan |
| `Get` — 10,000 entries | 2.2 ms | Encode + scan |

Encoding dominates (~460µs for n-gram, ~5ms for MiniLM). The linear scan adds
~67µs per 1,000 entries.

### Benchmark: xordb vs GPTCache

75-query dataset with 4 categories: 40 true semantic matches, 15 true negatives
(completely unrelated), 10 hard negatives (same topic, different question), and
10 edge cases (abbreviations, indirect phrasing). Both Docker containers run on
the same machine, same dataset, same rules.

|  | **xordb (n-gram)** | **xordb (MiniLM)** | **GPTCache** |
|---|---|---|---|
| **Accuracy** | 64.0% (48/75) | **86.7% (65/75)** | 81.3% (61/75) |
| **True positives** | 32 | **42** | 43 |
| **True negatives** | 16 | **23** | 18 |
| **False positives** | 9 | **2** | 7 |
| **False negatives** | 18 | 8 | 7 |
| **Avg latency** | **536µs** | 18.8ms | 285.4ms |
| **Heap (reported)** | **1.21 MB** | 23.17 MB | 6.32 MB * |
| **RSS (actual)** | **6.38 MB** | 192.64 MB † | 320.00 MB |
| **Dependencies** | **0** | onnxruntime + model | gptcache, faiss, onnxruntime, numpy |

\* GPTCache heap measured via Python `tracemalloc`, which does not capture
FAISS/ONNX/SQLite C++ allocations. RSS reflects the true cost.

† MiniLM RSS includes the ONNX Runtime engine (~170 MB) and the loaded model
weights. This is a fixed one-time cost — it does not grow with cache size. The
Go heap (23 MB) is the projector's hyperplane matrix + tokenizer vocabulary.

**Category breakdown:**

|  | **xordb (n-gram)** | **xordb (MiniLM)** | **GPTCache** |
|---|---|---|---|
| match (40) | 28/40 | **39/40** | **39/40** |
| neg (15) | 13/15 | **15/15** | **15/15** |
| hard-neg (10) | 3/10 | **8/10** | 3/10 |
| edge (10) | 4/10 | 3/10 | **4/10** |

Key takeaways:

- **MiniLM wins on accuracy** — 86.7% vs GPTCache's 81.3%, with only 2 false
  positives vs GPTCache's 7. It correctly rejects "speed of light" vs "speed of
  sound" while GPTCache doesn't.
- **N-gram wins on speed** — 536µs/query with zero dependencies and 6 MB RSS.
  If your paraphrases share words, this is all you need.
- **GPTCache is 15x slower** than MiniLM per query (285ms vs 19ms), largely due
  to Python overhead and FAISS index lookups.
- **RSS tells the real memory story.** GPTCache reports 6.3 MB via Python
  `tracemalloc`, but the actual process RSS is **320 MB** — FAISS, ONNX Runtime,
  SQLite, and numpy all allocate outside Python's tracked heap. xordb n-gram uses
  6.4 MB total. xordb MiniLM uses 193 MB (mostly the ONNX engine, fixed cost).
- **All three struggle with edge cases** — abbreviations like "ML" → "machine
  learning" and indirect phrasing like "why do apples fall" → "gravity" are hard
  for any embedding model at this size.

Reproduce it yourself:

```bash
bash benchmarks/run_comparison.sh
```

---

## Project structure

```
xordb/
├── xdb.go                Public API: New, NewWithEncoder, Options, Stats
├── xdb_test.go
│
├── hdc/
│   ├── vector.go         Vector type: Bundle, Bind, Similarity, Permute
│   ├── random.go         Seeded random hypervector generation
│   ├── encoder.go        Encoder interface + NGramEncoder
│   ├── pool.go           sync.Pool buffer recycling for encode ops
│   └── normalize.go      Text normalization
│
├── cache/
│   ├── cache.go          Cache: Set, Get, Delete, LRU eviction
│   └── cache_test.go
│
├── embed/                        ← separate Go module (xordb/embed)
│   ├── encoder.go                MiniLMEncoder: ONNX inference + projection
│   ├── projection.go             Random hyperplane LSH (float32 → binary)
│   ├── tokenizer.go              BERT WordPiece tokenizer
│   ├── vocab.go                  Embedded vocabulary (30k tokens, ~227KB)
│   ├── cmd/xordb-model/main.go   Model download CLI
│   └── go.mod                    Depends on onnxruntime_go + xordb
│
├── cmd/demo/
│   └── main.go           Interactive demo
│
└── go.mod                Zero dependencies
```

The core `xordb` module has **zero dependencies**. The `embed/` module is a
separate Go module that adds `onnxruntime_go` for local ML inference. You only
pull in the dependency if you import `xordb/embed`.

---

## Known limitations

**N-gram encoder is not transformer-quality.** Character n-grams capture surface
similarity. "capital of India" vs "capital of Nepal" score ~0.77 because they
share most characters. The default threshold of 0.82 avoids false positives but
also misses looser paraphrases. Use the MiniLM encoder for real semantic matching.

**Linear scan.** The cache does a full scan on every `Get`. At 10,000 entries
this is ~2ms — fine for most caching use cases. LSH indexing is planned for
larger scales.

**MiniLM adds binary size.** The ONNX model is ~90MB, downloaded separately.
The `onnxruntime` shared library must be available on the system.

**Encoding uses a buffer pool.** The n-gram encoder recycles `[]uint64` word
buffers and `[]int32` count buffers via `sync.Pool`, with in-place vector
operations (`permuteInto`, `bindInto`, `bundleInto`). Steady-state encodes
allocate ~5 KB/op instead of ~140 KB/op. The pool warms up after the first few
calls.

---

## Threshold guidance

| Threshold | Behaviour |
|-----------|-----------|
| `0.90 – 1.0` | Exact and near-exact matches only |
| `0.82` (default) | Conservative — low false-positive risk |
| `0.70 – 0.80` | Good balance for MiniLM encoder |
| `0.65 – 0.75` | Captures most paraphrases with n-gram encoder |
| `< 0.60` | Too permissive for production |

---

## Running tests

```bash
# Core library (zero deps)
go test ./...

# Embed module
cd embed && go test ./...

# With race detector
go test -race ./...

# Benchmarks
go test -bench=. -benchmem ./...

# Integration tests (requires ONNX model)
cd embed && go test -tags integration -v ./...

# Demo
go run ./cmd/demo/
go run ./cmd/demo/ -repl
```

---

## Roadmap

- [x] MiniLM local embeddings (`xordb/embed` module)
- [x] Buffer pool for n-gram encoder (`sync.Pool` recycling)
- [x] Docker benchmark suite (xordb vs GPTCache)
- [x] Custom encoder interface (`NewWithEncoder`)
- [x] WordPiece tokenizer (pure Go, zero deps)
- [x] Binary projection via random hyperplane LSH
- [x] Model management CLI (`xordb-model`)
- [ ] TTL expiration (auto-evict stale entries)
- [ ] Disk persistence (gob → flatbuffers upgrade path)
- [ ] LSH indexing for sub-linear lookup at scale
- [ ] HTTP sidecar mode
- [ ] SIMD assembly (`VPAND`, `VPOPCNTQ`)
- [ ] Additional model support (Nomic Embed, Arctic)
- [ ] Batch encoding API
