package hdc_test

import (
	"strings"
	"testing"
	"xordb/hdc"
)

// ── normalize (exported for testing via a thin wrapper) ───────────────────────
// normalize is unexported; we test it indirectly through Encode behaviour.

func TestEncode_CaseNormalization(t *testing.T) {
	enc := hdc.NewNGramEncoder(hdc.DefaultConfig())
	a := enc.Encode("Hello World")
	b := enc.Encode("hello world")
	if hdc.Similarity(a, b) != 1.0 {
		t.Fatal("case variants must produce identical vectors")
	}
}

func TestEncode_WhitespaceCollapse(t *testing.T) {
	enc := hdc.NewNGramEncoder(hdc.DefaultConfig())
	a := enc.Encode("hello   world")
	b := enc.Encode("hello world")
	if hdc.Similarity(a, b) != 1.0 {
		t.Fatal("whitespace variants must produce identical vectors")
	}
}

func TestEncode_StripPunctuation(t *testing.T) {
	cfg := hdc.DefaultConfig()
	cfg.StripPunctuation = true
	enc := hdc.NewNGramEncoder(cfg)
	a := enc.Encode("hello, world!")
	b := enc.Encode("hello world")
	if hdc.Similarity(a, b) != 1.0 {
		t.Fatal("stripped punctuation must produce identical vector to clean text")
	}
}

func TestEncode_Deterministic(t *testing.T) {
	enc := hdc.NewNGramEncoder(hdc.DefaultConfig())
	text := "the quick brown fox jumps over the lazy dog"
	a := enc.Encode(text)
	b := enc.Encode(text)
	if hdc.Similarity(a, b) != 1.0 {
		t.Fatal("Encode must be deterministic")
	}
}

func TestEncode_EmptyString(t *testing.T) {
	enc := hdc.NewNGramEncoder(hdc.DefaultConfig())
	v := enc.Encode("")
	zero := hdc.New(hdc.DefaultConfig().Dims)
	if hdc.Similarity(v, zero) != 1.0 {
		t.Fatal("empty input must return zero vector")
	}
}

func TestEncode_SingleRune(t *testing.T) {
	enc := hdc.NewNGramEncoder(hdc.DefaultConfig())
	v := enc.Encode("a")
	if v.Dims() != hdc.DefaultConfig().Dims {
		t.Fatalf("expected dims %d, got %d", hdc.DefaultConfig().Dims, v.Dims())
	}
}

// ── semantic similarity ordering ─────────────────────────────────────────────

// TestEncode_SimilarHigherThanUnrelated verifies the encoder produces
// meaningful similarity gradients, not just random noise.
func TestEncode_SimilarHigherThanUnrelated(t *testing.T) {
	enc := hdc.NewNGramEncoder(hdc.DefaultConfig())

	base := enc.Encode("what is the capital of india")
	rephrase := enc.Encode("capital city of india")
	unrelated := enc.Encode("how do you bake a chocolate cake")

	simRephrase := hdc.Similarity(base, rephrase)
	simUnrelated := hdc.Similarity(base, unrelated)

	if simRephrase <= simUnrelated {
		t.Fatalf(
			"rephrase (%.4f) should be more similar than unrelated (%.4f)",
			simRephrase, simUnrelated,
		)
	}
}

func TestEncode_TypoResistance(t *testing.T) {
	enc := hdc.NewNGramEncoder(hdc.DefaultConfig())
	a := enc.Encode("colour")
	b := enc.Encode("color")
	s := hdc.Similarity(a, b)
	if s < 0.65 {
		t.Fatalf("typo variants should be highly similar, got %.4f", s)
	}
}

func TestEncode_SharedPrefix(t *testing.T) {
	enc := hdc.NewNGramEncoder(hdc.DefaultConfig())
	a := enc.Encode("what is the capital of india")
	b := enc.Encode("what is the capital of nepal")
	c := enc.Encode("how do you make pasta carbonara")

	simSameTemplate := hdc.Similarity(a, b)
	simUnrelated := hdc.Similarity(a, c)

	// Both should be detectable; template similarity is known to be high for n-gram HDC.
	// We only assert that unrelated text is less similar than the template variant.
	if simUnrelated >= simSameTemplate {
		t.Logf("note: template similarity (%.4f) vs unrelated (%.4f)", simSameTemplate, simUnrelated)
		t.Fatal("unrelated text must be less similar than same-template text")
	}
}

// ── multi-sentence (paragraph) encoding ──────────────────────────────────────

func TestEncode_MultiSentence_SimilarToSingleSentence(t *testing.T) {
	enc := hdc.NewNGramEncoder(hdc.DefaultConfig())

	single := enc.Encode("the cat sat on the mat")
	multi := enc.Encode("the cat sat on the mat. it was a warm afternoon.")

	s := hdc.Similarity(single, multi)
	if s < 0.60 {
		t.Fatalf("multi-sentence with shared content should retain similarity, got %.4f", s)
	}
}

func TestEncode_SentenceSplitOnDelimiters(t *testing.T) {
	enc := hdc.NewNGramEncoder(hdc.DefaultConfig())

	a := enc.Encode("first sentence. second sentence.")
	b := enc.Encode("first sentence\nsecond sentence")

	s := hdc.Similarity(a, b)
	if s < 0.70 {
		t.Fatalf("period and newline delimiters should produce similar vectors, got %.4f", s)
	}
}

// ── long text (chunked encoding) ─────────────────────────────────────────────

func TestEncode_LongText_Deterministic(t *testing.T) {
	enc := hdc.NewNGramEncoder(hdc.DefaultConfig())
	long := strings.Repeat("the quick brown fox jumps over the lazy dog ", 10)
	a := enc.Encode(long)
	b := enc.Encode(long)
	if hdc.Similarity(a, b) != 1.0 {
		t.Fatal("long text encoding must be deterministic")
	}
}

func TestEncode_LongText_SimilarToSelf(t *testing.T) {
	enc := hdc.NewNGramEncoder(hdc.DefaultConfig())

	short := "the quick brown fox jumps over the lazy dog"
	long := strings.Repeat(short+" ", 6) // >200 chars, triggers chunking

	vShort := enc.Encode(short)
	vLong := enc.Encode(long)

	s := hdc.Similarity(vShort, vLong)
	if s < 0.55 {
		t.Fatalf("long repetition of phrase should retain similarity to short phrase, got %.4f", s)
	}
}

// ── Unicode ───────────────────────────────────────────────────────────────────

func TestEncode_Unicode_Deterministic(t *testing.T) {
	enc := hdc.NewNGramEncoder(hdc.DefaultConfig())
	a := enc.Encode("日本語のテキスト")
	b := enc.Encode("日本語のテキスト")
	if hdc.Similarity(a, b) != 1.0 {
		t.Fatal("Unicode text must encode deterministically")
	}
}

func TestEncode_Unicode_DifferentScripts(t *testing.T) {
	enc := hdc.NewNGramEncoder(hdc.DefaultConfig())
	latin := enc.Encode("hello world")
	kanji := enc.Encode("日本語のテキスト")
	assertNearHalf(t, "different scripts", hdc.Similarity(latin, kanji))
}

// ── NGramSize variants ────────────────────────────────────────────────────────

func TestEncode_NGramSize1(t *testing.T) {
	cfg := hdc.DefaultConfig()
	cfg.NGramSize = 1
	enc := hdc.NewNGramEncoder(cfg)
	a := enc.Encode("abc")
	b := enc.Encode("abc")
	if hdc.Similarity(a, b) != 1.0 {
		t.Fatal("unigram encoder must be deterministic")
	}
}

func TestEncode_NGramSize4(t *testing.T) {
	cfg := hdc.DefaultConfig()
	cfg.NGramSize = 4
	enc := hdc.NewNGramEncoder(cfg)
	a := enc.Encode("hello")
	b := enc.Encode("hello")
	if hdc.Similarity(a, b) != 1.0 {
		t.Fatal("4-gram encoder must be deterministic")
	}
}

// ── Encoder interface satisfaction ───────────────────────────────────────────

func TestNGramEncoder_ImplementsEncoder(t *testing.T) {
	var _ hdc.Encoder = hdc.NewNGramEncoder(hdc.DefaultConfig())
}

// ── Custom seed isolation ─────────────────────────────────────────────────────

func TestEncode_DifferentSeeds_QuasiOrthogonal(t *testing.T) {
	cfg1 := hdc.DefaultConfig()
	cfg1.Seed = 1
	cfg2 := hdc.DefaultConfig()
	cfg2.Seed = 2

	enc1 := hdc.NewNGramEncoder(cfg1)
	enc2 := hdc.NewNGramEncoder(cfg2)

	text := "hello world"
	assertNearHalf(t, "same text, different seed", hdc.Similarity(enc1.Encode(text), enc2.Encode(text)))
}

// ── Concurrency ───────────────────────────────────────────────────────────────

func TestEncode_ConcurrentSafe(t *testing.T) {
	enc := hdc.NewNGramEncoder(hdc.DefaultConfig())
	texts := []string{
		"the quick brown fox",
		"hello world",
		"semantic caching with hdc",
		"hyperdimensional computing",
	}

	done := make(chan struct{}, len(texts)*4)
	for i := 0; i < 4; i++ {
		for _, txt := range texts {
			go func(s string) {
				enc.Encode(s)
				done <- struct{}{}
			}(txt)
		}
	}
	for i := 0; i < len(texts)*4; i++ {
		<-done
	}
}

// ── Config validation ─────────────────────────────────────────────────────────

func TestNewNGramEncoder_InvalidDims_Panics(t *testing.T) {
	assertPanics(t, "Dims=0", func() {
		cfg := hdc.DefaultConfig()
		cfg.Dims = 0
		hdc.NewNGramEncoder(cfg)
	})
}

func TestNewNGramEncoder_InvalidNGramSize_Panics(t *testing.T) {
	assertPanics(t, "NGramSize=0", func() {
		cfg := hdc.DefaultConfig()
		cfg.NGramSize = 0
		hdc.NewNGramEncoder(cfg)
	})
}

func TestNewNGramEncoder_InvalidChunkSize_Panics(t *testing.T) {
	assertPanics(t, "ChunkSize=1", func() {
		cfg := hdc.DefaultConfig()
		cfg.ChunkSize = 1
		hdc.NewNGramEncoder(cfg)
	})
}

func TestNewNGramEncoder_InvalidLongTextThresh_Panics(t *testing.T) {
	assertPanics(t, "LongTextThresh=0", func() {
		cfg := hdc.DefaultConfig()
		cfg.LongTextThresh = 0
		hdc.NewNGramEncoder(cfg)
	})
}

// ── Unicode whitespace ────────────────────────────────────────────────────────

func TestEncode_NonBreakingSpace_CollapsedLikeSpace(t *testing.T) {
	enc := hdc.NewNGramEncoder(hdc.DefaultConfig())
	// U+00A0 non-breaking space should be treated as whitespace
	a := enc.Encode("hello\u00A0world")
	b := enc.Encode("hello world")
	if hdc.Similarity(a, b) != 1.0 {
		t.Fatal("non-breaking space must collapse to regular space")
	}
}

// ── Benchmarks ────────────────────────────────────────────────────────────────

func BenchmarkEncode_Short(b *testing.B) {
	enc := hdc.NewNGramEncoder(hdc.DefaultConfig())
	text := "what is the capital of india"
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		enc.Encode(text)
	}
}

func BenchmarkEncode_Medium(b *testing.B) {
	enc := hdc.NewNGramEncoder(hdc.DefaultConfig())
	text := strings.Repeat("the quick brown fox jumps over the lazy dog ", 4) // ~180 chars
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		enc.Encode(text)
	}
}

func BenchmarkEncode_Long(b *testing.B) {
	enc := hdc.NewNGramEncoder(hdc.DefaultConfig())
	text := strings.Repeat("the quick brown fox jumps over the lazy dog ", 12) // ~530 chars, triggers chunking
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		enc.Encode(text)
	}
}

func BenchmarkEncode_WarmSymbolTable(b *testing.B) {
	enc := hdc.NewNGramEncoder(hdc.DefaultConfig())
	text := "what is the capital of india"
	enc.Encode(text) // warm up symbol table
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		enc.Encode(text)
	}
}

// BenchmarkEncode_WarmPool exercises the encoder with both the symbol table
// and the buffer pool warmed up, showing the steady-state allocation profile.
func BenchmarkEncode_WarmPool(b *testing.B) {
	enc := hdc.NewNGramEncoder(hdc.DefaultConfig())
	text := "what is the capital of india"
	// Warm up both the symbol table and the buffer pool.
	for i := 0; i < 10; i++ {
		enc.Encode(text)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		enc.Encode(text)
	}
}
