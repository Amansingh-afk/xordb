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

**SQLite for similarity, embed a semantic index in any Go app.**  
*Offline. Private. Zero cloud costs.*

A lightweight similarity store built on [hdc-go](https://github.com/Amansingh-afk/hdc-go)
(Hyperdimensional Computing). Written in Go.

License: Apache-2.0 (see [LICENSE](LICENSE)).

---

## Why xordb

Vector databases are powerful but heavy, Pinecone, Weaviate, Qdrant all need
infrastructure. Sometimes you just need to store things and find similar ones,
without spinning up a server.

xordb is the **embeddable** option. Import it, store entries, query by
similarity. No network, no server, no external dependencies. It runs in-process
with 11 MB RSS.

```
"What is the capital of India?"
"capital city of india"
"India's capital?"
```

Three different strings. Same intent. xordb matches them all.

Think of it as SQLite for similarity. 

---

## How it works

xordb uses [hdc-go](https://github.com/Amansingh-afk/hdc-go) to encode text
into **hypervectors**: binary arrays where meaning is distributed across
every bit. Similarity is measured with Hamming distance: XOR + popcount,
nanoseconds per comparison.

With the MiniLM encoder, lookups are **two-stage**: a binary Hamming scan
selects the top-K candidates, then an exact cosine pass over stored
int8-quantized embeddings picks the winner. The hit threshold applies to
the cosine score, so accuracy matches a full float vector store while the
scan stays bitwise.

On top of that, xordb adds everything you need for a production store: LSH
indexing for sub-linear lookups, LRU eviction, TTL expiry, atomic disk
persistence with CRC checksums, and a thread-safe API.

xordb ships with two encoding modes:

| Mode | How it works | Quality | Dependencies |
|------|-------------|---------|-------------|
| **Built-in** (n-gram HDC) | Character n-grams with positional binding | Good for surface-level similarity | Zero |
| **MiniLM** (`xordb/embed`) | Local ONNX transformer model → binary projection | Near-OpenAI semantic quality | `onnxruntime_go` + model file |

Pick the one that fits your use case. The core library stays zero-dependency
either way.

---

## Use cases

### LLM response caching

Your agent answers "what is the capital of India?" once, and every rephrase
hits the store instead of burning another API call.

```go
db.Set("what is the capital of india", llmResponse)

v, ok, _ := db.Get("capital city of india")
// ok=true, v=llmResponse, saved one LLM call
```

### Intent classification / routing

Pre-populate with known intents. Incoming queries get routed in sub-millisecond
without a model. Runnable demo: [`examples/intent-routing`](examples/intent-routing).

```go
db.Set("check my order status", "order_status")
db.Set("I want a refund", "refund")
db.Set("talk to a human", "escalate")

intent, ok, _ := db.Get("where is my package")  // → "order_status"
intent, ok, _ = db.Get("give me my money back")  // → "refund"
```

### Deduplication

Find near-duplicates across datasets. Surface-level or semantic, depending on
the encoder.

```go
db.Set("annual report 2024 Q3 financial results", docID1)

_, isDup, sim := db.Get("2024 Q3 annual financial report results")
// isDup=true, sim≈0.85, duplicate detected
```

### FAQ matching

Pre-populate with FAQ pairs. User questions get matched semantically without
a full NLP pipeline.

```go
db.Set("what is your refund policy", "30-day money-back guarantee")
db.Set("how do I contact support", "Email support@example.com")

db.Get("can I get a refund")       // hits refund policy
db.Get("how to reach your team")   // hits contact support
```

### RAG pre-filtering

Use xordb as a fast first pass before expensive vector search. Narrow 1M
candidates to 1K in microseconds, then re-rank with a full model.

```go
db.Set("how do I reset my password", retrievedChunks)

db.Get("password reset instructions")
// Hit, skip the entire retrieval pipeline
```

### Edge / offline

xordb runs entirely in-process. No Redis, no server, no network.

- **Mobile apps**: on-device similarity search
- **IoT / embedded**: 11 MB RSS, runs on Raspberry Pi
- **Privacy-sensitive apps**: queries never leave the process
- **Air-gapped environments**: zero external dependencies at runtime

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
// ok=false, unrelated, correctly rejected
```

### Mode 2: Semantic quality (MiniLM encoder)

Real semantic understanding. "author of ramayana" matches "who wrote ramayana"
even though the words barely overlap. Runs a quantized MiniLM-L6-v2 model
locally via ONNX Runtime, no API calls, no cloud, no per-query cost.

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
// ok=true, real semantic match, not just string overlap
```

### How to choose

| | Built-in (n-gram) | MiniLM (`xordb/embed`) |
|---|---|---|
| **Dependencies** | Zero | `onnxruntime_go` + model file |
| **Semantic quality** | Surface-level (word overlap) | Near-OpenAI (MTEB benchmarks) |
| **"capital of india" ↔ "india's capital"** | hits (shared n-grams) | hits (semantic match) |
| **"who wrote ramayana" ↔ "author of ramayana"** | misses (no word overlap) | hits (semantic match) |
| **Encode latency** | ~500us | ~5ms |
| **Binary size** | ~2MB | ~2MB + 90MB model |
| **Best for** | Typo tolerance, dedup, intent routing, edge | Semantic search, RAG, chatbot FAQ |

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
| `WithThreshold(t)` | `0.75` | Minimum similarity for a hit, in `(0, 1]`. Cosine with the MiniLM encoder, Hamming similarity with n-gram. |
| `WithCapacity(n)` | `1024` | Max entries. Oldest evicted when exceeded (LRU). |
| `WithNGramSize(n)` | `3` | Character n-gram window. |
| `WithSeed(s)` | `0` | Encoder seed. DBs with different seeds are incompatible. |
| `WithStripPunctuation(v)` | `false` | Strip punctuation before encoding. |
| `WithTTL(d)` | `0` (no expiry) | Default time-to-live for entries. Expired entries are lazily reaped on next `Get`. |
| `WithLSH(bool)` | auto | Enable/disable LSH indexing. Auto-enabled when capacity ≥ 256. |
| `WithLSHParams(k, l)` | auto | Override auto-computed LSH parameters (k=bits sampled, l=tables). |
| `WithLSHFallback(bool)` | `true` | Fall back to linear scan on LSH miss. Preserves exact semantics. |
| `WithRerankK(k)` | `16` | Rerank window for encoders with float embeddings (MiniLM). `-1` disables. |

**With custom encoder (e.g. MiniLM):**

```go
db := xordb.NewWithEncoder(enc, opts ...Option)
```

Accepts any [`hdc.Encoder`](https://github.com/Amansingh-afk/hdc-go) implementation.
Encoding options (`WithDims`, `WithNGramSize`, etc.) are ignored since the
encoder controls those. If the encoder also implements `Embed`/`Project`
(like the MiniLM encoder), two-stage cosine rerank is enabled automatically.

### MiniLM encoder options

```go
enc, err := embed.NewMiniLMEncoder(
    embed.WithModelPath("/path/to/model.onnx"),  // default: auto-detect
    embed.WithMaxSeqLen(128),                      // default: 128
    embed.WithBinaryDims(1024),                    // default: 1024
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
atomic, data goes to a `.tmp` file, fsynced, then renamed into place. Expired
entries are stripped on save.

```go
db.Load(path string) error
```
Load a previously saved snapshot. Expired entries are skipped. Merges into the
live cache, existing entries survive, duplicate keys get overwritten by the
snapshot. Returns a wrapped `os.ErrNotExist` if the file is missing, so you can
safely call `Load` on first run and check with `errors.Is`.

```go
db.Save("cache.xrdb")

// Later, or in a new process:
db.Load("cache.xrdb")
```

The binary format includes a CRC-32 checksum over the entry payload. Corrupted
files are rejected on load. Values are serialized as JSON internally, structs,
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

### HDC primitives ([hdc-go](https://github.com/Amansingh-afk/hdc-go))

| Operation | Time | Notes |
|-----------|------|-------|
| `Similarity` | **67 ns** | Pure POPCNT, no alloc |
| `Bind` | 365 ns | XOR + allocation |
| `Permute` | 520 ns | Clone + cyclic shift |
| `Bundle(10)` | 141 µs | Majority vote, 10 vecs × 10k bits |

### Cache lookups

| Operation | Time (linear) | Time (LSH) | Notes |
|-----------|---------------|------------|-------|
| `Set` (n-gram) | 532 µs | - | Encode + insert |
| `Get` (100 entries) | 460 µs | - | Encode + scan |
| `Get` (1,000 entries) | 1.2 ms | 1.2 ms | ~Even at 1K |
| `Get` (10,000 entries) | 3.1 ms | **1.6 ms** | **~1.9× faster with LSH** |

Encoding dominates (~460µs for n-gram, ~5ms for MiniLM). Linear scan adds
~67µs per 1,000 entries. LSH benefit grows with entry count and data diversity.

### Benchmark: xordb vs GPTCache

444-query dataset from [Quora Question Pairs](https://huggingface.co/datasets/SetFit/qqp)
with 3 categories: 310 true semantic matches (paraphrases), 21 true negatives
(completely unrelated), and 113 hard negatives (same topic, different question).
Both Docker containers run on the same machine, same dataset, same rules. All
systems use their default thresholds (xordb: 0.75).

|  | **xordb (n-gram)** | **xordb (MiniLM)** ‡ | **GPTCache** |
|---|---|---|---|
| **Precision** | 83.7% (41/49) | **81.4% (275/338)** | 75.1% (307/409) |
| **Recall** | 13.2% (41/310) | 88.7% (275/310) | **99.0% (307/310)** |
| **F1 Score** | 22.8% | **84.9%** | 85.4% |
| **FP Rate** | **6.0% (8/134)** | 47.0% (63/134) | 76.1% (102/134) |
| **False neg** | 269 | 35 | **3** |
| **Avg latency** | **1.1ms** | 19.1ms | 283.6ms |
| **Heap (reported)** | **2.25 MB** | 22.67 MB | 6.36 MB * |
| **RSS (actual)** | **10.80 MB** | 197.59 MB † | 323.06 MB |
| **Dependencies** | **0** | onnxruntime + model | gptcache, faiss, onnxruntime, numpy |

\* GPTCache heap measured via Python `tracemalloc`, which does not capture
FAISS/ONNX/SQLite C++ allocations. RSS reflects the true cost.

† MiniLM RSS includes the ONNX Runtime engine (~170 MB) and the loaded model
weights. This is a fixed one-time cost, it does not grow with cache size. The
Go heap (23 MB) is the projector's hyperplane matrix + tokenizer vocabulary.

‡ Measured with two-stage cosine rerank at the default 0.75 threshold
(cosine). F1 matches brute-force float cosine on this dataset. Raise the
threshold for higher precision (0.90 → 91.3%).

**Category breakdown:**

|  | **xordb (n-gram)** | **xordb (MiniLM)** ‡ | **GPTCache** |
|---|---|---|---|
| match (310) | 41/310 | 275/310 | **307/310** |
| neg (21) | **21/21** | 17/21 | 14/21 |
| hard-neg (113) | **105/113** | 54/113 | 18/113 |

Key takeaways:

- **GPTCache has a 76% false positive rate.** It returns a cached answer for
  76% of queries that *should not* match. For a semantic cache, this means
  serving wrong answers most of the time on non-matching queries. High recall
  (99%) is meaningless if precision is low.
- **MiniLM with rerank matches GPTCache's F1 (84.9% vs 85.4%)** at 15x lower
  latency and 38% less RSS, with better precision (81% vs 75%). The pipeline
  scores within rounding of brute-force float cosine; remaining false
  positives on hard negatives are the embedding model's limit.
- **N-gram wins on precision and speed**: 84% precision, only 6% FP rate,
  1.1ms/query with zero dependencies and 11 MB RSS. Recall is low (13%)
  because character n-grams can't match paraphrases with different words,
  but when it says "hit", it's almost always right.
- **GPTCache is 15x slower** than MiniLM per query (284ms vs 19ms), largely
  due to Python overhead and FAISS index lookups.
- **RSS tells the real memory story.** GPTCache reports 6 MB via Python
  `tracemalloc`, but the actual process RSS is **323 MB**: FAISS, ONNX
  Runtime, SQLite, and numpy all allocate outside Python's tracked heap.
  xordb n-gram uses 11 MB total. xordb MiniLM uses 198 MB (mostly the ONNX
  engine, fixed cost).

Reproduce it yourself:

```bash
bash benchmarks/run_comparison.sh
```

### Benchmark: xordb vs chromem-go

[chromem-go](https://github.com/philippgille/chromem-go) is the closest
comparable: a pure-Go embeddable vector store using brute-force float cosine.
Both stores were fed identical MiniLM embeddings. Store-side cost is measured
with precomputed embeddings so the (identical) ONNX inference cost doesn't
mask the difference. 5,000 docs, 200 queries:

|  | **xordb** | **chromem-go** |
|---|---|---|
| **Accuracy (444 Quora pairs)** | F1 85.7% @ 0.70 | F1 85.7% @ 0.70 |
| **Query (store-side)** | **2.0 ms** | 7.5 ms |
| **Memory per entry** | **1.5 KB** | 2.1 KB |
| **Ingest (store-side)** | 1.9s total (projection) | **48 ms** total |
| **Query (end-to-end incl. embed)** | **24.2 ms** | 29.7 ms |

Accuracy is identical because the rerank stage scores with the same cosine.
The binary scan makes queries ~3.7x faster store-side, and entries are ~26%
smaller (128-byte binary vector + 384-byte int8 embedding vs 1.5 KB of
float32). The trade-off is a one-time projection cost per insert (~0.4 ms);
end-to-end ingest is dominated by the embedding model for both stores.

Reproduce: `TestComparison_Accuracy` / `TestComparison_Perf` in `benchmarks/`.

---

## Architecture

```
hdc-go/                           ← standalone HDC primitives (zero deps)
│   Vector, Bundle, Bind, Similarity, Permute
│   NGramEncoder, Projector (bring your own embeddings)
│
xordb/                            ← similarity store (this repo)
├── xdb.go                Public API: New, NewWithEncoder, Options, Stats
├── xdb_test.go
│
├── cache/
│   ├── cache.go          Store: Set, Get, Delete, LRU eviction
│   ├── lsh.go            LSH index: bit-sampling hash, insert/remove/query
│   ├── persist.go        Snapshot / LoadSnapshot (in-memory)
│   └── binary.go         Binary encode/decode (.xrdb format, CRC-32)
│
├── embed/                        ← separate Go module (xordb/embed)
│   ├── encoder.go                MiniLMEncoder: ONNX inference + projection
│   ├── tokenizer.go              BERT WordPiece tokenizer
│   ├── vocab.go                  Embedded vocabulary (30k tokens, ~227KB)
│   ├── cmd/xordb-model/main.go   Model download CLI
│   └── go.mod                    Depends on onnxruntime_go + hdc-go
│
├── cmd/demo/
│   └── main.go           Interactive demo
│
└── go.mod                Depends on hdc-go (zero transitive deps)
```

HDC primitives live in [hdc-go](https://github.com/Amansingh-afk/hdc-go), a
standalone library with zero dependencies. xordb adds the storage layer: LSH
indexing, LRU eviction, TTL expiry, persistence, and the thread-safe API.

The `embed/` module is a separate Go module that adds `onnxruntime_go` for local
ML inference. You only pull in that dependency if you import `xordb/embed`.

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
similarity 0.90. With fallback enabled (default), you get exact semantics, and
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
share most characters. The default threshold of 0.75 balances recall and precision but
also misses looser paraphrases. Use the MiniLM encoder for real semantic matching.

**Linear scan by default at small scale.** Below 256 entries, the cache does a
full scan on every `Get` (~2ms at 10K entries). At 256+ entries, LSH indexing
kicks in automatically for sub-linear lookups. With LSH enabled and fallback on
(default), exact semantics are preserved, LSH narrows candidates first, and a
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
| `0.75` (default) | Good balance for both encoders |
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
- [x] Extract HDC primitives into standalone [hdc-go](https://github.com/Amansingh-afk/hdc-go) library
- [ ] Improve recall to 85%+ (multi-hash HDC, dimensionality tuning)
- [ ] Batch similarity search API
- [ ] Namespace / collection support
- [ ] HTTP sidecar mode
- [ ] SIMD assembly (`VPAND`, `VPOPCNTQ`)
- [ ] Benchmark against FAISS, Annoy, ScaNN
