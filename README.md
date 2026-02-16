# xordb

A semantic cache powered by Hyperdimensional Computing (HDC).
Written in Go. Zero dependencies. No server required.

---

## The problem

Every app that calls an LLM pays per request. Most real-world usage repeats the
same questions in slightly different words:

```
"What is the capital of India?"
"India's capital city?"
"Tell me the capital city of India"
```

A normal cache misses all three — they are different strings.
xordb hits all three — it understands they mean the same thing.

---

## How it works

Text is encoded into a **hypervector**: a 10,000-bit array where meaning is
distributed across every bit. Similarity between two hypervectors is measured
with Hamming distance — a bitwise operation that runs in nanoseconds.

Three primitives power everything:

| Operation | Purpose | Implementation |
|-----------|---------|----------------|
| `Bundle`  | Combine concepts ("A or B") | majority vote, bit-by-bit |
| `Bind`    | Associate concepts ("A means B") | XOR (self-inverse) |
| `Similarity` | How alike are two things? | popcount / Hamming |

Text is encoded using character n-grams with positional binding — no external
ML model, no tokenizer, no network call. The encoder is deterministic and runs
entirely in-process.

---

## Current status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Core HDC engine (`hdc/`) | Done |
| 2 | Text encoder (`hdc/encoder.go`) | Done |
| 3 | Semantic cache (`cache/`) | Done |
| 4 | Public API (`xdb.go`) | Done |
| 5 | Demo CLI + benchmarks | Done |
| 6 | Persistence, float vectors, HTTP mode | Stretch goal |

---

## Quick start

```go
import "xordb"

db := xordb.New()

db.Set("what is the capital of india", "Delhi")

// Exact hit
v, ok, sim := db.Get("what is the capital of india")
// ok=true, sim=1.0, v="Delhi"

// Semantic hit (lower threshold needed for n-gram HDC)
db2 := xordb.New(xordb.WithThreshold(0.65))
db2.Set("what is the capital of india", "Delhi")

v, ok, sim = db2.Get("capital city of india")
// ok=true, sim≈0.72, v="Delhi"

// Miss
_, ok, _ = db.Get("how do you bake a chocolate cake")
// ok=false
```

---

## Use Cases

xordb is a general-purpose semantic cache that works anywhere you need to find similar things, not just exact matches. Here are practical use cases across many domains:

### LLM & AI Applications

**LLM Response Caching**
```go
// Cache expensive LLM API calls
db.Set("what is the capital of india?", "Delhi")
db.Get("capital city of india") // ✅ Cache hit - saves API cost
```

**Chatbot FAQ Systems**
```go
// Pre-populate with FAQs, match user questions semantically
db.Set("what is your refund policy?", "30-day money-back guarantee")
db.Get("refund policy") // ✅ Matches
```

**Translation Memory**
```go
// Reuse translations for similar source text
db.Set("Hello world", "Hola mundo")
db.Get("Hello, world!") // ✅ Cache hit
```

### Software Development

**Code Similarity Detection**
```go
db.Set("function add(a, b) { return a + b; }", "file1.js")
db.Get("function add(x, y) { return x + y; }") // ✅ Detects similar code
```

**Test Case Deduplication**
```go
// Avoid running similar tests
db.Set("test user login with email", testResult)
if _, hit, _ := db.Get("test login with email address"); hit {
    skipTest() // Similar test already ran
}
```

**Log Pattern Matching**
```go
// Route similar log entries to same handlers
db.Set("ERROR: database connection failed", "db_error")
db.Get("ERROR: db connection failure") // ✅ Matches pattern
```

### Data Processing

**Document Deduplication**
```go
// Detect duplicate or near-duplicate documents
db.Set(documentContent, documentID)
if _, hit, _ := db.Get(newDocument); hit {
    flagAsDuplicate()
}
```

**Streaming Deduplication**
```go
// "Have I seen this before?" in O(1) per item
// Perfect for log processing, data pipelines
if _, hit, _ := db.Get(dataItem); !hit {
    process(dataItem)
    db.Set(dataItem, true)
}
```

**Content Clustering**
```go
// Group similar content together
for _, item := range items {
    if cluster, hit, _ := db.Get(item); hit {
        addToCluster(item, cluster)
    } else {
        db.Set(item, newClusterID())
    }
}
```

### Web & API Development

**API Response Caching**
```go
// Cache any expensive API call, not just LLMs
db.Set("GET /api/users?filter=active", apiResponse)
db.Get("GET /api/users?status=active") // ✅ Cache hit
```

**Search Query Expansion**
```go
// Store query → results mappings
db.Set("best restaurants NYC", results1)
db.Get("NYC dining recommendations") // ✅ Returns cached results
```

**Session & State Caching**
```go
// Cache user sessions, feature flags, configs by context
db.Set("user_context_v1", sessionData)
db.Get("user_context_v2") // ✅ Similar context
```

### E-commerce & Recommendations

**Product Search**
```go
// "laptop" matches "notebook computer"
db.Set("laptop", productResults)
db.Get("notebook computer") // ✅ Cache hit
```

**Review Deduplication**
```go
// Detect duplicate reviews
db.Set(reviewText, reviewID)
db.Get(similarReviewText) // ✅ Flags duplicate
```

**Recommendation Caching**
```go
// Cache recommendations for similar user profiles
db.Set(userProfile, recommendations)
db.Get(similarProfile) // ✅ Reuse recommendations
```

### Security & Monitoring

**Spam/Abuse Detection**
```go
// Store known spam patterns
db.Set("click here for free money!!!", "spam")
db.Get("CLICK HERE FOR FREE MONEY") // ✅ Detects variation
```

**Anomaly Detection**
```go
// Bundle known-good events, flag vectors far from bundle
goodEvents := bundle(event1, event2, event3)
if similarity(newEvent, goodEvents) < threshold {
    flagAsAnomaly()
}
```

**Fraud Detection**
```go
// Match similar transaction patterns
db.Set(transactionPattern, "fraud")
db.Get(similarPattern) // ✅ Flags potential fraud
```

### Healthcare & Legal

**Symptom Matching**
```go
// "headache" matches "head pain"
db.Set("headache", diagnosisPath)
db.Get("head pain") // ✅ Same diagnosis path
```

**Case Law Similarity**
```go
// Find similar legal cases
db.Set(caseDescription, caseID)
db.Get(similarCase) // ✅ Finds related cases
```

**Contract Clause Matching**
```go
// Match similar contract clauses
db.Set(clauseText, clauseID)
db.Get(similarClause) // ✅ Finds matches
```

### IoT & Edge Computing

**Sensor Data Deduplication**
```go
// Edge device: avoid sending duplicate readings
if _, hit, _ := db.Get(sensorReading); !hit {
    sendToCloud(sensorReading)
    db.Set(sensorReading, true)
}
```

**Device Configuration Caching**
```go
// Cache configs for similar device contexts
db.Set(deviceContext, config)
db.Get(similarContext) // ✅ Reuse config
```

### Gaming & Entertainment

**Cheat Detection**
```go
// Detect similar behavior patterns
db.Set(knownCheatPattern, "cheat")
db.Get(playerBehavior) // ✅ Flags suspicious behavior
```

**Player Behavior Analysis**
```go
// Group similar player behaviors
db.Set(playerBehavior, behaviorType)
db.Get(similarBehavior) // ✅ Classifies behavior
```

### Education

**Plagiarism Detection**
```go
// Detect similar essays/assignments
db.Set(submission1, "original")
db.Get(submission2) // ✅ Flags potential plagiarism
```

**Question Bank Matching**
```go
// "what is photosynthesis?" matches variations
db.Set("what is photosynthesis?", answer)
db.Get("explain photosynthesis") // ✅ Cache hit
```

### Email & Communication

**Email Thread Grouping**
```go
// Group similar email subjects
db.Set("Re: Project update", threadID)
db.Get("RE: project update") // ✅ Same thread
```

**Message Deduplication**
```go
// Avoid processing duplicate messages
db.Set(messageContent, messageID)
db.Get(similarMessage) // ✅ Flags duplicate
```

### Database & Query Optimization

**SQL Query Caching**
```go
// Cache SQL queries (even with different formatting)
db.Set("SELECT * FROM users WHERE active = 1", queryResult)
db.Get("select * from users where active=1") // ✅ Cache hit
```

**Query Result Caching**
```go
// Cache expensive query results
db.Set(queryDescription, results)
db.Get(similarQuery) // ✅ Reuse results
```

### Future Applications (with HDC primitives)

**Knowledge Graphs**
```go
// Bind relationships: "Delhi is-capital-of India"
// Future: Associative store with Bind operations
```

**Few-shot Classification**
```go
// Bundle examples per class, classify by nearest bundle
catExamples := bundle("cat image 1", "cat image 2", ...)
dogExamples := bundle("dog image 1", "dog image 2", ...)
// Classify by nearest bundle
```

---

## Why xordb Works Across Domains

1. **Text-based keys**: Anything representable as a string works
2. **Semantic similarity**: Finds meaning, not exact matches
3. **Fast**: Sub-millisecond lookups
4. **Zero dependencies**: Works anywhere Go runs
5. **Embeddable**: No server needed
6. **Thread-safe**: Works in concurrent systems

**The pattern**: If you can represent it as a string and want to find similar strings, xordb can help. The HDC engine is domain-agnostic—it's just fast semantic similarity search.

---

## API

### Creating a DB

```go
db := xordb.New(opts ...Option)
```

| Option | Default | Description |
|--------|---------|-------------|
| `WithDims(n int)` | `10000` | Hypervector dimension. Higher = more accurate, more memory. |
| `WithThreshold(t float64)` | `0.82` | Minimum similarity for a cache hit. Range: `(0, 1]`. |
| `WithCapacity(n int)` | `1024` | Max entries. Oldest entry is evicted when exceeded (LRU). |
| `WithNGramSize(n int)` | `3` | Character n-gram window. Larger = more precise, less typo-tolerant. |
| `WithSeed(s uint64)` | `0` | Encoder namespace seed. DBs with different seeds are incompatible. |
| `WithStripPunctuation(v bool)` | `false` | Strip punctuation during normalization. Useful for NL queries. |

`New` panics if any option is invalid (e.g. `Capacity=0`, `Threshold=0`, `Dims=0`).

### Methods

```go
db.Set(key string, value any)
```
Store any value under a string key. If the exact key exists, update it and
promote it to most-recently-used. Encodes the key outside the lock.

```go
db.Get(key string) (value any, hit bool, similarity float64)
```
Return the value stored under the most similar key at or above the threshold.
Returns `(nil, false, 0)` on a miss. The matched entry is promoted to MRU on a hit.

```go
db.Delete(key string) bool
```
Remove the entry with the **exact** key string. Returns true if found.
Note: Delete is exact-match only. Pass the same string used in `Set`.

```go
db.Len() int
```
Current number of cached entries.

```go
db.Stats() xordb.Stats
```
Point-in-time snapshot:

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

## Test commands

```bash
# Run all tests
go test ./...

# Run tests with verbose output
go test -v ./...

# Run tests for a specific package
go test -v ./hdc/
go test -v ./cache/
go test -v .          # root package (xordb)

# Run benchmarks
go test -bench=. -benchmem ./...
go test -bench=. -benchmem ./hdc/
go test -bench=. -benchmem ./cache/
go test -bench=. -benchmem .

# Race detector
go test -race ./...

# Run the demo
go run ./cmd/demo/

# Run the demo with interactive REPL
go run ./cmd/demo/ -repl
```

---

## Performance

Measured on Intel i7-13620H (16 threads). All allocations are heap — a future
`BindInto` / object-pool path would reduce this significantly.

### HDC primitives (`hdc/`)

| Operation | Time | Allocs | Notes |
|-----------|------|--------|-------|
| `Similarity` | **67 ns** | 0 | Pure POPCNT, no alloc |
| `Bind` | 365 ns | 1 × 1.25 KB | XOR + allocation |
| `Permute` | 520 ns | 1 × 1.25 KB | Clone + cyclic shift |
| `Bundle(10)` | 141 µs | 2 | Majority vote, 10 vecs × 10k bits |
| `Random` | 14 µs | 2 | Seeded PRNG + alloc |

### Encoder (`hdc/`)

| Input | Time | Notes |
|-------|------|-------|
| Short (~29 chars) — cold | 510 µs | Symbol table populated on first call |
| Short (~29 chars) — warm | 521 µs | Symbol lookups are cached; encoding dominates |
| Medium (~180 chars) | 2.6 ms | Linear with n-gram count |
| Long (~530 chars) | 14 ms | Chunked encoding, many Bundle calls |

Encoding is allocation-heavy (145 allocs / 258 KB for a short query). The root
cause is every `Bind` and `Permute` allocating a new `[]uint64`. A scratch-buffer
pool would cut this roughly 10×.

### Cache (`cache/`, `xordb`)

| Operation | Time | Notes |
|-----------|------|-------|
| `DB.Set` | 532 µs | Encode (446 µs) + lock + list insert |
| `DB.Get` — 100 entries | 460 µs | Encode + 100 × 67 ns scan |
| `DB.Get` — 1000 entries | 820 µs | Encode + 1000 × 67 ns scan |
| `DB.Get` — 10000 entries | 2.2 ms | Encode + 10000 × 67 ns scan |

The encoding (≈460 µs) dominates for any cache size. The linear scan adds ≈67 µs
per 1000 entries — at 10,000 entries the scan contributes ≈670 µs (total ~2.2 ms).

---

## Project structure

```
xordb/
├── xdb.go              Public API: DB, Option funcs, Stats
├── xdb_test.go
│
├── hdc/
│   ├── vector.go       Vector type: Bundle, Bind, Similarity, Permute, Clone
│   ├── random.go       Seeded random hypervector generation
│   ├── encoder.go      Encoder interface, NGramEncoder, Config, symbolTable
│   ├── normalize.go    normalize(), splitSentences()
│   ├── hdc_test.go
│   └── encoder_test.go
│
├── cache/
│   ├── cache.go        Cache: Set, Get, Delete, Len, Stats, LRU eviction
│   └── cache_test.go
│
└── cmd/
    └── demo/
        └── main.go     HDC primitives demo
```

---

## Known limitations

**N-gram HDC is not transformer-quality.**
Character n-grams capture surface similarity, not deep semantics. Queries that
share sentence structure but differ in meaning (e.g. "capital of India" vs
"capital of Nepal") score closer together than they ideally should (measured:
~0.77 vs ~0.72 for an India paraphrase at default dims=10000). The default
threshold of 0.82 avoids most false positives but also misses looser paraphrases.

**Practical threshold guidance:**

| Threshold | Behaviour |
|-----------|-----------|
| `0.95 – 1.0` | Exact and near-exact matches only |
| `0.82` (default) | Conservative — low false-positive risk |
| `0.65 – 0.75` | Captures most paraphrases; higher false-positive risk |
| `< 0.65` | Too permissive for production use |

**The fix (Phase 6):** accept external float embeddings (OpenAI, BERT, etc.)
projected to binary via random projection. The HDC engine becomes a fast binary
index; the embedding model provides the semantic quality.

**Encoding is allocation-heavy.**
Each `Bind` and `Permute` call allocates a 1.25 KB `[]uint64`. For a 30-char
query this is ~145 allocations / ~258 KB. An object-pool / in-place API is
planned.

---

## Stretch goals

- Float vector input (external embeddings → binary projection)
- Disk persistence (gob)
- HTTP sidecar mode
- SIMD assembly (`VPAND`, `VPOPCNTQ`)
