# XDB — Roadmap & Concept Guide

> A blazingly fast Semantic Cache powered by Hyperdimensional Computing (HDC).
> Written in Go. Zero dependencies. No server required.

---

## What Problem Does XDB Solve?

Every app that calls an LLM (ChatGPT, Claude, Gemini) pays per request.
Most real-world usage repeats the same questions in slightly different words:

```
"What is the capital of India?"
"India's capital city?"
"Tell me the capital city of India"
```

A normal cache would miss all three — they're different strings.
XDB hits all three — it understands they *mean the same thing*.

This is a **Semantic Cache**: cache by meaning, not by exact text.

---

## The Two Core Ideas

### 1. Hyperdimensional Computing (HDC)

**What it is:**
A way to represent *any piece of information* as a giant array of bits —
typically 10,000 bits — called a **hypervector**.

**The key insight:**
In very high dimensions, random vectors are nearly orthogonal to each other.
This means:
- Two completely unrelated things → hypervectors that are ~50% different
- Two similar things → hypervectors that are < 20% different
- The same thing encoded twice → nearly identical hypervectors

This property is called *quasi-orthogonality* and it's what makes similarity
search possible with just bitwise operations.

**Three operations — that's it:**

| Operation | What it does | How |
|-----------|-------------|-----|
| `Bundle`  | Combine multiple concepts ("A or B or C") | majority vote bit-by-bit |
| `Bind`    | Associate two concepts ("A means B") | XOR |
| `Similarity` | How alike are two things? | Hamming distance / popcount |

**Why this is novel:**
Traditional vector DBs store 768 or 1536 float32 values per vector (~6KB).
HDC stores 10,000 bits per vector (1.25KB) and similarity is **bitwise** —
the fastest possible operation on any CPU.

---

### 2. How Text Gets Encoded into a Hypervector

We use **character n-grams** — overlapping windows of characters.

Example: encoding `"hello"` with n=3:
```
window 1: "hel"  → bind(H[h], H[e], H[l])  → v1
window 2: "ell"  → bind(H[e], H[l], H[l])  → v2
window 3: "llo"  → bind(H[l], H[l], H[o])  → v3
final:    bundle(v1, v2, v3)               → hypervector for "hello"
```

Where `H[x]` is a **fixed random hypervector** assigned to character `x` at
startup. The same character always maps to the same hypervector.

**Why n-grams?**
- Typo-resistant: "colour" and "color" share most n-grams
- Language-agnostic: works on any text without tokenization
- No ML model needed: encoding is pure deterministic math

---

## Why Is XDB Fast?

### 1. Bitwise Operations
Hamming distance = count the differing bits between two vectors.
Modern CPUs have a `POPCNT` instruction that counts 64 bits in a single cycle.
For a 10,000-bit vector, similarity takes ~157 `POPCNT` calls. That's nanoseconds.

### 2. No Float Math
Traditional cosine similarity requires:
- 768 multiplications (float32)
- 768 additions
- A square root

HDC similarity requires:
- 157 XOR operations
- 157 POPCNT operations
- 1 division

Roughly **5-10x fewer operations** per comparison.

### 3. Cache-Friendly Memory Layout
Each hypervector is a flat `[]uint64` — 157 contiguous 64-bit words.
No pointers. No heap allocations per comparison. CPU prefetcher loves this.

### 4. Linear Scan is Viable
With ~nanosecond comparisons, a linear scan of 100,000 cached entries
takes ~15ms. For most semantic caches (thousands of entries), this is
under 1ms. No index structure needed at small-medium scale.

### 5. Future: SIMD Acceleration
Go doesn't expose SIMD directly, but the compiler autovectorizes
simple loop patterns. Future: assembly stubs for AVX2 to process
256 bits per cycle instead of 64.

---

## Why Is XDB Unique?

| Feature | Traditional Vector DB | XDB |
|---------|----------------------|-----|
| Vector type | float32 (dense) | uint64 bitpacked |
| Similarity | cosine / dot product | Hamming (bitwise) |
| Encoding | external ML model required | built-in n-gram HDC |
| Index | HNSW / IVF (complex) | linear scan (simple, fast enough) |
| Dependencies | many | zero |
| Deployment | server / sidecar | embedded library |
| Memory per entry | ~6KB (768 float32) | ~1.25KB (10K bits) |
| Supports binding | no | yes (key-value association) |
| Supports bundling | no | yes (concept sets) |

**No production library exists today that:**
- Uses HDC as a semantic cache engine
- Is embedded (no server)
- Has zero dependencies
- Is written in Go

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                    xdb (public API)                  │
│  New() / Set() / Get() / Delete() / Stats()          │
└────────────────────┬─────────────────────────────────┘
                     │
          ┌──────────▼──────────┐
          │    cache.Cache      │
          │  threshold-based    │
          │  similarity lookup  │
          │  LRU eviction       │
          └──────────┬──────────┘
                     │
       ┌─────────────▼────────────┐
       │       hdc.Encoder        │
       │  text → hypervector      │
       │  n-gram + bundle + bind  │
       └─────────────┬────────────┘
                     │
       ┌─────────────▼────────────┐
       │       hdc.Vector         │
       │  []uint64 bitpacked      │
       │  Bundle / Bind / Hamming │
       └──────────────────────────┘
```

---

## Development Phases

### Phase 1 — Core HDC Engine (`hdc/`) ✅
- [x] `Vector` type: `[]uint64`, dimension-agnostic
- [x] `Random(dims, seed)` — generate a random hypervector
- [x] `Bundle(vecs...)` — majority vote
- [x] `Bind(a, b)` — XOR
- [x] `Similarity(a, b) float64` — normalized Hamming (0=opposite, 1=identical)
- [x] `Clone()`, `Permute()` for positional encoding
- [x] `sync.Pool` buffer recycling + in-place ops (`permuteInto`, `bindInto`, `bundleInto`)

### Phase 2 — Text Encoder (`hdc/encoder.go`) ✅
- [x] `Encoder` as an **interface** — allows plugging in external embeddings
- [x] `NGramEncoder` — default implementation, symbol table (rune → Vector)
- [x] `Normalize(input string) string` — lowercase, collapse whitespace, optional punctuation strip
- [x] `Encode(text string) Vector` — normalize → sliding n-gram window
- [x] Configurable n (default: 3), Unicode-aware
- [x] **Long text fix**: sliding window (128-char chunks, 50% overlap) and bundle results
- [x] **Paragraph fix**: hierarchical encoding — encode each sentence, then bundle sentence vectors
- [x] Sentence splitter (split on `.`, `?`, `!`, `\n`)

**What the key (query) can be — everything is a string:**
| Input type | Works? | Notes |
|---|---|---|
| Plain text | Great | Primary use case |
| JSON string | Structural | Similar shape/keys score high |
| Code snippet | Good | N-grams work well for code similarity |
| Math (verbal) | OK | "integral of x squared" works |
| Math (notation) | Poor | "∫x²dx" — different chars, known limitation |
| Text files | Good | Pass content as string |
| Binary files | No | Meaningless output — caller must extract text first |

**The contract**: key = `string` (encoded to hypervector). Value = `any` (returned as-is on hit).

### Phase 3 — Semantic Cache (`cache/`) ✅
- [x] `Cache` struct: stores `(hypervector, value, key_string, timestamp)`
- [x] `Set(key string, value any)`
- [x] `Get(key string) (any, bool, float64)` — returns value + similarity score
- [x] Configurable similarity threshold (default: 0.82)
- [x] LRU eviction when capacity exceeded
- [x] Thread-safe (RWMutex)

### Phase 4 — Public API (`xdb.go`) ✅
- [x] `New(opts ...Option) *DB`
- [x] `Options`: dims, threshold, capacity, n-gram size
- [x] `DB.Set / Get / Delete / Len / Stats()`
- [x] `Stats`: hit rate, avg similarity on hits, total entries
- [x] `NewWithEncoder(enc, opts...)` — custom encoder support

### Phase 5 — Demo & Benchmarks ✅
- [x] `cmd/demo/` — interactive CLI to show cache hits
- [x] `BenchmarkEncode` — how fast is text → hypervector?
- [x] `BenchmarkGet` — how fast is a cache lookup?
- [x] `BenchmarkGet10k` — 10,000 entries in cache
- [x] Compare similarity scores for related vs unrelated queries
- [x] Docker benchmark suite (xordb n-gram + MiniLM vs GPTCache)

### Phase 6 — MiniLM Local Embeddings (`embed/`) ✅
- [x] `MiniLMEncoder` — ONNX inference + binary projection via random hyperplane LSH
- [x] Pure Go WordPiece tokenizer (30k vocab, embedded)
- [x] Model management CLI (`xordb-model download / info / path`)
- [x] Separate Go module — core stays zero-dependency

### Phase 7 — Next Up
- [ ] TTL expiration — auto-evict stale entries (critical for caching correctness)
- [ ] Disk persistence — gob for simplicity, flatbuffers as upgrade path
- [ ] LSH indexing — sub-linear lookup for large caches (10k+ entries)
- [ ] HTTP sidecar mode — REST API for polyglot use
- [ ] SIMD assembly (`VPAND`, `VPOPCNTQ`) — 4x throughput on AVX2
- [ ] Additional model support (Nomic Embed, Arctic)
- [ ] Batch encoding API
- [ ] Associative store: `Bind("delhi", "india")` then query by association

> Note: billion-scale, distributed sharding, and LangChain adapters are out of scope.
> Ship something real and embedded first.

---

## Key Concepts Glossary

**Hypervector**
A vector of 10,000 bits. Meaning is distributed across all bits — no single
bit "means" anything. Robustness comes from the sheer dimensionality.

**Bundling**
Combining multiple hypervectors into one that "resembles all of them."
Used to represent sets or categories.
`bundle(cat, dog, fish)` → a vector similar to all three.

**Binding**
Creating a new hypervector that represents the *association* of two things.
XOR is used because it's self-inverse: `bind(bind(a, b), b) == a`.
Used to encode key-value relationships.

**Hamming Distance**
The number of bit positions where two vectors differ.
Normalized: `hamming(a, b) / dims` → range [0, 1].
`1.0` = identical. `0.0` = opposite. `~0.5` = unrelated.

**Similarity Threshold**
The minimum similarity score to count as a cache hit.
Too high (0.99): only exact matches → many misses.
Too low (0.60): too many false positives → wrong cached values returned.
Sweet spot: **0.80 – 0.85** for natural language queries.

**N-gram**
A sliding window of N characters over a string.
`"hello"` with n=3 → `["hel", "ell", "llo"]`.
Captures local structure without needing a tokenizer.

**Quasi-orthogonality**
In high dimensions, almost all randomly chosen vectors are nearly
perpendicular (similarity ≈ 0.5). This means the chance of a false positive
from a random unrelated vector is astronomically low.

---

## What Success Looks Like

```
$ xdb demo

Query:  "what is the capital of India?"
Action: SET → cached
Score:  1.000 (exact)

Query:  "India's capital city?"
Action: GET → cache HIT  ✓
Score:  0.847 (similar enough)
Value:  "Delhi"

Query:  "what is the capital of Nepal?"
Action: GET → cache MISS ✗
Score:  0.612 (not similar enough)

Query:  "capital city of india"
Action: GET → cache HIT  ✓
Score:  0.831
Value:  "Delhi"
```

---

## Why This Matters Beyond the Cache

HDC is a general-purpose computing paradigm. Once the engine exists:
- **Knowledge graphs**: bind("Delhi", "is-capital-of", "India") as a single vector
- **Anomaly detection**: bundle known-good events, flag vectors far from the bundle
- **Few-shot classification**: bundle examples per class, classify by nearest bundle
- **Streaming dedup**: seen-before detection in O(1) per item

XDB starts as a cache. The engine is a foundation.

---

*XDB — because meaning should be a first-class citizen in data storage.*
