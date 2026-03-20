# xordb

[![CI](https://github.com/Amansingh-afk/xordb/actions/workflows/ci.yml/badge.svg)](https://github.com/Amansingh-afk/xordb/actions/workflows/ci.yml)

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

License: Apache-2.0 (see [LICENSE](LICENSE)).

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
import "github.com/Amansingh-afk/xordb"

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
go run github.com/Amansingh-afk/xordb/embed/cmd/xordb-model download
```

```go
import (
    "github.com/Amansingh-afk/xordb"
    "github.com/Amansingh-afk/xordb/embed"
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
| `WithTTL(d)` | `0` (no expiry) | Default time-to-live for entries. Expired entries are lazily reaped on next `Get`. |
| `WithLSH(bool)` | auto | Enable/disable LSH indexing. Auto-enabled when capacity ≥ 256. |
| `WithLSHParams(k, l)` | auto | Override auto-computed LSH parameters (k=bits sampled, l=tables). |
| `WithLSHFallback(bool)` | `true` | Fall back to linear scan on LSH miss. Preserves exact semantics. |

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
promote to most-recently-used. Uses the cache's default TTL.

```go
db.SetWithTTL(key string, value any, ttl time.Duration)
```
Store with a per-entry TTL that overrides the cache default. A TTL of zero
means the entry never expires.

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
    Entries       int
    Hits          uint64
    Misses        uint64
    Sets          uint64
    Expired       uint64
    HitRate       float64
    AvgSimOnHit   float64
    LSHCandidates uint64   // total candidates evaluated via LSH across all Gets
    LSHFallbacks  uint64   // number of times LSH missed and fell back to linear scan
}
```

### Persistence

```go
db.Save(path string) error
```
Write the cache to disk. Uses a custom binary format (`.xrdb`). The write is
atomic — data goes to a `.tmp` file, fsynced, then renamed into place. Expired
entries are stripped on save.

```go
db.Load(path string) error
```
Load a previously saved snapshot. Expired entries are skipped. Merges into the
live cache — existing entries survive, duplicate keys get overwritten by the
snapshot. Returns a wrapped `os.ErrNotExist` if the file is missing, so you can
safely call `Load` on first run and check with `errors.Is`.

```go
db.Save("cache.xrdb")

// Later, or in a new process:
db.Load("cache.xrdb")
```

The binary format includes a CRC-32 checksum over the entry payload. Corrupted
files are rejected on load. Values are serialized as JSON internally — structs,
maps, slices, and primitives all work without registration. The only caveat:
values come back as their JSON-decoded types (e.g. `int` becomes `float64`,
structs become `map[string]any`).

---

## Model management

The MiniLM encoder needs a one-time model download (~90MB). The `xordb-model`
CLI handles this:

```bash
# Download the model
go run github.com/Amansingh-afk/xordb/embed/cmd/xordb-model download

# Check status
go run github.com/Amansingh-afk/xordb/embed/cmd/xordb-model info

# Print model path
go run github.com/Amansingh-afk/xordb/embed/cmd/xordb-model path
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

| Operation | Time (linear) | Time (LSH) | Notes |
|-----------|---------------|------------|-------|
| `Set` (n-gram) | 532 µs | — | Encode + insert |
| `Get` — 100 entries | 460 µs | — | Encode + scan |
| `Get` — 1,000 entries | 1.2 ms | 1.2 ms | ~Even at 1K |
| `Get` — 10,000 entries | 3.1 ms | **1.6 ms** | **~1.9× faster with LSH** |

Encoding dominates (~460µs for n-gram, ~5ms for MiniLM). Linear scan adds
~67µs per 1,000 entries. LSH benefit grows with entry count and data diversity.

### Benchmark: xordb vs GPTCache

444-query dataset from [Quora Question Pairs](https://huggingface.co/datasets/SetFit/qqp)
with 3 categories: 310 true semantic matches (paraphrases), 21 true negatives
(completely unrelated), and 113 hard negatives (same topic, different question).
Both Docker containers run on the same machine, same dataset, same rules. All
systems use their default thresholds (xordb: 0.82).

|  | **xordb (n-gram)** | **xordb (MiniLM)** | **GPTCache** |
|---|---|---|---|
| **Precision** | 83.7% (41/49) | **86.1% (199/231)** | 75.1% (307/409) |
| **Recall** | 13.2% (41/310) | 64.2% (199/310) | **99.0% (307/310)** |
| **F1 Score** | 22.8% | **73.6%** | 85.4% |
| **FP Rate** | **6.0% (8/134)** | 23.9% (32/134) | 76.1% (102/134) |
| **False neg** | 269 | 111 | **3** |
| **Avg latency** | **1.1ms** | 19.1ms | 283.6ms |
| **Heap (reported)** | **2.25 MB** | 22.67 MB | 6.36 MB * |
| **RSS (actual)** | **10.80 MB** | 197.59 MB † | 323.06 MB |
| **Dependencies** | **0** | onnxruntime + model | gptcache, faiss, onnxruntime, numpy |

\* GPTCache heap measured via Python `tracemalloc`, which does not capture
FAISS/ONNX/SQLite C++ allocations. RSS reflects the true cost.

† MiniLM RSS includes the ONNX Runtime engine (~170 MB) and the loaded model
weights. This is a fixed one-time cost — it does not grow with cache size. The
Go heap (23 MB) is the projector's hyperplane matrix + tokenizer vocabulary.

**Category breakdown:**

|  | **xordb (n-gram)** | **xordb (MiniLM)** | **GPTCache** |
|---|---|---|---|
| match (310) | 41/310 | 199/310 | **307/310** |
| neg (21) | **21/21** | 20/21 | 14/21 |
| hard-neg (113) | **105/113** | 82/113 | 18/113 |

Key takeaways:

- **GPTCache has a 76% false positive rate.** It returns a cached answer for
  76% of queries that *should not* match. For a semantic cache, this means
  serving wrong answers most of the time on non-matching queries. High recall
  (99%) is meaningless if precision is low.
- **MiniLM has the best F1 balance** — 73.6% F1 with 86% precision. It
  correctly rejects hard negatives (82/113) while still finding most true
  matches (199/310). False positive rate of 24% is manageable and tunable
  via threshold.
- **N-gram wins on precision and speed** — 84% precision, only 6% FP rate,
  1.1ms/query with zero dependencies and 11 MB RSS. Recall is low (13%)
  because character n-grams can't match paraphrases with different words,
  but when it says "hit", it's almost always right.
- **GPTCache is 15x slower** than MiniLM per query (284ms vs 19ms), largely
  due to Python overhead and FAISS index lookups.
- **RSS tells the real memory story.** GPTCache reports 6 MB via Python
  `tracemalloc`, but the actual process RSS is **323 MB** — FAISS, ONNX
  Runtime, SQLite, and numpy all allocate outside Python's tracked heap.
  xordb n-gram uses 11 MB total. xordb MiniLM uses 198 MB (mostly the ONNX
  engine, fixed cost).

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
│   ├── lsh.go            LSH index: bit-sampling hash, insert/remove/query
│   ├── persist.go        Snapshot / LoadSnapshot (in-memory)
│   ├── binary.go         Binary encode/decode (.xrdb format)
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

## LSH indexing

At scale, linear scanning every entry on `Get` becomes the bottleneck. xordb
uses **Locality-Sensitive Hashing** (bit-sampling) to reduce lookups to
sub-linear time.

### How it works

Each LSH table samples `k` random bit positions from the hypervector and packs
them into a bucket key. Vectors that agree on all `k` bits land in the same
bucket. With `L` independent tables, the probability of finding a similar entry
is boosted to near-certainty for high-similarity pairs.

- **Auto-enabled** when capacity ≥ 256 (disable with `WithLSH(false)`)
- **Parameters auto-tuned** from threshold (override with `WithLSHParams(k, l)`)
- **Fallback** to full linear scan on LSH miss (default: on, preserves exact semantics)

### Tuning

| Parameter | Effect |
|-----------|--------|
| Higher `k` | Fewer candidates per bucket → faster but lower recall |
| Higher `L` | More tables → higher recall but more memory |
| `WithLSHFallback(false)` | Skip linear scan on miss → faster but may miss edge cases |

The auto-computed defaults target ~74% recall at threshold and ~98% at
similarity 0.90. With fallback enabled (default), you get exact semantics —
LSH just accelerates the common case.

```go
// Explicit LSH control
db := xordb.New(
    xordb.WithCapacity(10000),
    xordb.WithLSH(true),               // force enable
    xordb.WithLSHParams(14, 20),        // 14 bits, 20 tables
    xordb.WithLSHFallback(true),        // fall back to scan on miss
)
```

---

## Known limitations

**N-gram encoder is not transformer-quality.** Character n-grams capture surface
similarity. "capital of India" vs "capital of Nepal" score ~0.77 because they
share most characters. The default threshold of 0.82 avoids false positives but
also misses looser paraphrases. Use the MiniLM encoder for real semantic matching.

**Linear scan by default at small scale.** Below 256 entries, the cache does a
full scan on every `Get` (~2ms at 10K entries). At 256+ entries, LSH indexing
kicks in automatically for sub-linear lookups. With LSH enabled and fallback on
(default), exact semantics are preserved — LSH narrows candidates first, and a
full scan runs only if LSH misses. See the [LSH section](#lsh-indexing) below.

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
- [x] TTL expiration (lazy eviction on `Get`, global + per-entry override)
- [x] Disk persistence (custom binary format, CRC-32, atomic writes)
- [x] LSH indexing for sub-linear lookup at scale
- [ ] HTTP sidecar mode
- [ ] SIMD assembly (`VPAND`, `VPOPCNTQ`)
- [ ] Additional model support (Nomic Embed, Arctic)
- [ ] Batch encoding API
