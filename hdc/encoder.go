package hdc

import (
	"strings"
	"sync"
)

// Encoder converts a string to a hypervector.
// The interface allows swapping in external embedding models (Phase 6).
type Encoder interface {
	Encode(text string) Vector
}

// Config holds parameters for an NGramEncoder.
type Config struct {
	Dims             int    // hypervector dimension (default 10000)
	NGramSize        int    // sliding window width in runes (default 3)
	StripPunctuation bool   // remove punctuation during normalization
	LongTextThresh   int    // rune count above which chunked encoding is used (default 200)
	ChunkSize        int    // rune count per chunk, 50% overlap (default 128)
	Seed             uint64 // namespace seed; same seed → same symbol table
}

// DefaultConfig returns production-ready defaults.
func DefaultConfig() Config {
	return Config{
		Dims:           10000,
		NGramSize:      3,
		LongTextThresh: 200,
		ChunkSize:      128,
	}
}

// NGramEncoder implements Encoder using character n-grams and HDC.
// It is safe for concurrent use.
type NGramEncoder struct {
	cfg Config
	sym symbolTable
}

// NewNGramEncoder creates an NGramEncoder with the given Config.
// Panics if any required field is zero or invalid.
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
	}
}

// Encode returns a hypervector representing text.
// Empty text (after normalization) returns a zero vector.
func (e *NGramEncoder) Encode(text string) Vector {
	if text == "" {
		return New(e.cfg.Dims)
	}

	// Lowercase before splitting so sentence delimiters are reliably detected.
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
	return Bundle(vecs...)
}

// encodeRunes encodes a rune slice using a sliding n-gram window.
// Falls back to per-rune bundling when len(runes) < NGramSize.
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
		return Bundle(vecs...)
	}

	count := len(runes) - n + 1
	vecs := make([]Vector, count)
	for i := range vecs {
		vecs[i] = e.encodeWindow(runes[i : i+n])
	}
	return Bundle(vecs...)
}

// encodeWindow encodes a single n-gram window using position-sensitive binding.
// result = ρ⁰(h₀) XOR ρ¹(h₁) XOR … XOR ρⁿ⁻¹(hₙ₋₁)
// Permuting each symbol by its position ensures "hel" ≠ "lhe".
func (e *NGramEncoder) encodeWindow(runes []rune) Vector {
	result := e.sym.get(runes[0])
	for i := 1; i < len(runes); i++ {
		sym := e.sym.get(runes[i])
		for j := 0; j < i; j++ {
			sym = sym.Permute()
		}
		result = Bind(result, sym)
	}
	return result
}

// encodeChunked splits runes into overlapping 50% chunks and bundles the results.
// Tail chunks shorter than NGramSize are skipped because the preceding overlapping
// chunk already covers that content, and a sub-NGramSize chunk cannot form any
// n-gram — it would only contribute per-rune fallback noise to the bundle.
// The first chunk is always included regardless of length.
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
	return Bundle(vecs...)
}

// symbolTable is a thread-safe lazy map from rune to a deterministic random Vector.
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
	// Knuth multiplicative hash mixed with the encoder seed for namespace isolation.
	v = Random(t.dims, t.seed^uint64(r)*2654435761+1)
	t.table[r] = v
	return v
}
