package hdc

import (
	"strings"
	"sync"
)

// Encoder converts text to a hypervector. Implement this to plug in your own
// embedding model (see xordb/embed for MiniLM).
type Encoder interface {
	Encode(text string) Vector
}

// Config for the built-in n-gram encoder.
type Config struct {
	Dims             int
	NGramSize        int
	StripPunctuation bool
	LongTextThresh   int    // rune count above which chunked encoding kicks in
	ChunkSize        int    // rune count per chunk, 50% overlap
	Seed             uint64 // same seed = same symbol table
}

func DefaultConfig() Config {
	return Config{
		Dims:           10000,
		NGramSize:      3,
		LongTextThresh: 200,
		ChunkSize:      128,
	}
}

// NGramEncoder — character n-grams + HDC. Thread-safe.
// Uses pooled buffers internally to keep allocations low.
type NGramEncoder struct {
	cfg  Config
	sym  symbolTable
	pool *bufPool
}

func NewNGramEncoder(cfg Config) *NGramEncoder {
	switch {
	case cfg.Dims <= 0:
		panic("hdc: Config.Dims must be positive")
	case cfg.NGramSize <= 0:
		panic("hdc: Config.NGramSize must be positive")
	case cfg.ChunkSize < 2:
		panic("hdc: Config.ChunkSize must be >= 2 (stride = ChunkSize/2 must be non-zero)")
	case cfg.LongTextThresh <= 0:
		panic("hdc: Config.LongTextThresh must be positive")
	}
	return &NGramEncoder{
		cfg: cfg,
		sym: symbolTable{
			dims:  cfg.Dims,
			seed:  cfg.Seed,
			table: make(map[rune]Vector),
		},
		pool: newBufPool(cfg.Dims),
	}
}

// Encode returns a hypervector for the given text. Empty input → zero vector.
func (e *NGramEncoder) Encode(text string) Vector {
	if text == "" {
		return New(e.cfg.Dims)
	}

	sentences := splitSentences(strings.ToLower(text))
	vecs := make([]Vector, 0, len(sentences))
	for _, s := range sentences {
		s = normalizeSegment(s, e.cfg.StripPunctuation)
		if s == "" {
			continue
		}
		runes := []rune(s)
		var v Vector
		if len(runes) > e.cfg.LongTextThresh {
			v = e.encodeChunked(runes)
		} else {
			v = e.encodeRunes(runes)
		}
		vecs = append(vecs, v)
	}

	if len(vecs) == 0 {
		return New(e.cfg.Dims)
	}
	if len(vecs) == 1 {
		return vecs[0]
	}
	return e.bundlePooled(vecs)
}

// bundlePooled — majority vote using pooled scratch buffers.
// returned vector owns its data (safe to store).
func (e *NGramEncoder) bundlePooled(vecs []Vector) Vector {
	counts := e.pool.getCounts()
	dst := vectorFromBuf(e.cfg.Dims, e.pool.getWords())
	bundleInto(dst, counts, vecs)
	e.pool.putCounts(counts)
	// dst ka buffer pool mein wapas nahi jaayega — caller stores it
	return dst
}

func (e *NGramEncoder) encodeRunes(runes []rune) Vector {
	n := e.cfg.NGramSize
	if len(runes) < n {
		vecs := make([]Vector, len(runes))
		for i, r := range runes {
			vecs[i] = e.sym.get(r)
		}
		if len(vecs) == 0 {
			return New(e.cfg.Dims)
		}
		return e.bundlePooled(vecs)
	}

	count := len(runes) - n + 1

	windowBufs := make([][]uint64, count)
	vecs := make([]Vector, count)
	for i := range vecs {
		buf := e.pool.getWords()
		windowBufs[i] = buf
		vecs[i] = vectorFromBuf(e.cfg.Dims, buf)
		e.encodeWindowInto(vecs[i], runes[i:i+n])
	}

	result := e.bundlePooled(vecs)

	for _, buf := range windowBufs {
		e.pool.putWords(buf)
	}

	return result
}

// encodeWindowInto — position-sensitive binding for one n-gram window.
// result = ρ⁰(h₀) XOR ρ¹(h₁) XOR … XOR ρⁿ⁻¹(hₙ₋₁)
func (e *NGramEncoder) encodeWindowInto(dst Vector, runes []rune) {
	dims := e.cfg.Dims
	sym0 := e.sym.get(runes[0])
	copy(dst.data, sym0.data)

	if len(runes) == 1 {
		return
	}

	// ping-pong buffers for permutation
	scratchA := e.pool.getWords()
	scratchB := e.pool.getWords()
	defer e.pool.putWords(scratchA)
	defer e.pool.putWords(scratchB)

	permSrc := vectorFromBuf(dims, scratchA)
	permDst := vectorFromBuf(dims, scratchB)

	for i := 1; i < len(runes); i++ {
		sym := e.sym.get(runes[i])
		copy(permSrc.data, sym.data)
		for j := 0; j < i; j++ {
			permuteInto(permDst, permSrc)
			permSrc, permDst = permDst, permSrc
		}
		bindInto(dst, dst, permSrc)
	}
}

// encodeChunked — splits long text into overlapping 50% chunks and bundles.
func (e *NGramEncoder) encodeChunked(runes []rune) Vector {
	size := e.cfg.ChunkSize
	stride := size / 2
	var vecs []Vector
	for start := 0; start < len(runes); start += stride {
		end := start + size
		if end > len(runes) {
			end = len(runes)
		}
		chunk := runes[start:end]
		if len(chunk) >= e.cfg.NGramSize || len(vecs) == 0 {
			vecs = append(vecs, e.encodeRunes(chunk))
		}
		if end == len(runes) {
			break
		}
	}
	if len(vecs) == 0 {
		return New(e.cfg.Dims)
	}
	if len(vecs) == 1 {
		return vecs[0]
	}
	return e.bundlePooled(vecs)
}

// symbolTable — lazy rune → Vector map. Thread-safe.
type symbolTable struct {
	mu    sync.RWMutex
	dims  int
	seed  uint64
	table map[rune]Vector
}

func (t *symbolTable) get(r rune) Vector {
	t.mu.RLock()
	v, ok := t.table[r]
	t.mu.RUnlock()
	if ok {
		return v
	}

	t.mu.Lock()
	defer t.mu.Unlock()
	if v, ok = t.table[r]; ok {
		return v
	}
	v = Random(t.dims, t.seed^uint64(r)*2654435761+1)
	t.table[r] = v
	return v
}
