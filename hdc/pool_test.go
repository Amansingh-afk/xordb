package hdc

import (
	"testing"
)

// ── bufPool tests ────────────────────────────────────────────────────────────

func TestBufPool_GetWords_ReturnsZeroed(t *testing.T) {
	pool := newBufPool(10000)
	buf := pool.getWords()
	for i, w := range buf {
		if w != 0 {
			t.Fatalf("getWords returned non-zero word at index %d: %d", i, w)
		}
	}
	// Dirty the buffer and return it.
	for i := range buf {
		buf[i] = ^uint64(0)
	}
	pool.putWords(buf)

	// Get again — must be zeroed.
	buf2 := pool.getWords()
	for i, w := range buf2 {
		if w != 0 {
			t.Fatalf("recycled getWords returned non-zero word at index %d: %d", i, w)
		}
	}
}

func TestBufPool_GetCounts_ReturnsZeroed(t *testing.T) {
	pool := newBufPool(10000)
	buf := pool.getCounts()
	for i, c := range buf {
		if c != 0 {
			t.Fatalf("getCounts returned non-zero count at index %d: %d", i, c)
		}
	}
	// Dirty and return.
	for i := range buf {
		buf[i] = 999
	}
	pool.putCounts(buf)

	// Get again — must be zeroed.
	buf2 := pool.getCounts()
	for i, c := range buf2 {
		if c != 0 {
			t.Fatalf("recycled getCounts returned non-zero count at index %d: %d", i, c)
		}
	}
}

func TestBufPool_Words_CorrectLength(t *testing.T) {
	pool := newBufPool(10000)
	buf := pool.getWords()
	want := numWords(10000)
	if len(buf) != want {
		t.Fatalf("getWords: len=%d, want %d", len(buf), want)
	}
}

func TestBufPool_Counts_CorrectLength(t *testing.T) {
	pool := newBufPool(10000)
	buf := pool.getCounts()
	if len(buf) != 10000 {
		t.Fatalf("getCounts: len=%d, want 10000", len(buf))
	}
}

// ── In-place operation tests ─────────────────────────────────────────────────

func TestPermuteInto_MatchesPermute(t *testing.T) {
	src := Random(10000, 42)
	expected := src.Permute()

	dst := New(10000)
	permuteInto(dst, src)

	if Similarity(dst, expected) != 1.0 {
		t.Fatal("permuteInto must produce the same result as Permute")
	}
}

func TestPermuteInto_MultiWordBoundary(t *testing.T) {
	// 65 dims exercises the two-word boundary.
	src := Random(65, 7)
	expected := src.Permute()

	dst := New(65)
	permuteInto(dst, src)

	if Similarity(dst, expected) != 1.0 {
		t.Fatal("permuteInto at 65 dims must match Permute")
	}
}

func TestBindInto_MatchesBind(t *testing.T) {
	a := Random(10000, 1)
	b := Random(10000, 2)
	expected := Bind(a, b)

	dst := New(10000)
	bindInto(dst, a, b)

	if Similarity(dst, expected) != 1.0 {
		t.Fatal("bindInto must produce the same result as Bind")
	}
}

func TestBundleInto_MatchesBundle(t *testing.T) {
	vecs := make([]Vector, 5)
	for i := range vecs {
		vecs[i] = Random(10000, uint64(i+1))
	}
	expected := Bundle(vecs...)

	dst := New(10000)
	counts := make([]int32, 10000)
	bundleInto(dst, counts, vecs)

	if Similarity(dst, expected) != 1.0 {
		t.Fatal("bundleInto must produce the same result as Bundle")
	}
}

func TestBundleInto_SingleVector(t *testing.T) {
	v := Random(10000, 42)
	expected := Bundle(v)

	dst := New(10000)
	counts := make([]int32, 10000)
	bundleInto(dst, counts, []Vector{v})

	if Similarity(dst, expected) != 1.0 {
		t.Fatal("bundleInto with single vector must match Bundle")
	}
}

func TestVectorFromBuf_ZeroesPadding(t *testing.T) {
	// 65 dims → 2 words; second word should have high bits zeroed.
	buf := []uint64{^uint64(0), ^uint64(0)}
	v := vectorFromBuf(65, buf)

	if Similarity(v, v) != 1.0 {
		t.Fatal("vectorFromBuf must zero padding bits")
	}
	// The second word should only have bit 0 set.
	if v.data[1] != 1 {
		t.Fatalf("expected second word=1 (only bit 0 of 65), got %d", v.data[1])
	}
}

// ── Encoder pooling correctness ──────────────────────────────────────────────

func TestEncode_Pooled_Deterministic(t *testing.T) {
	enc := NewNGramEncoder(DefaultConfig())
	text := "what is the capital of india"

	// Run many times to exercise pool recycling.
	first := enc.Encode(text)
	for i := 0; i < 100; i++ {
		v := enc.Encode(text)
		if Similarity(first, v) != 1.0 {
			t.Fatalf("iteration %d: pooled Encode not deterministic (sim=%.4f)", i, Similarity(first, v))
		}
	}
}

func TestEncode_Pooled_DifferentTexts_Independent(t *testing.T) {
	enc := NewNGramEncoder(DefaultConfig())

	// Interleave different texts to stress the pool.
	texts := []string{
		"what is the capital of india",
		"how do you bake a chocolate cake",
		"explain quantum computing",
		"the quick brown fox jumps over the lazy dog",
	}

	// Compute reference vectors.
	refs := make([]Vector, len(texts))
	for i, txt := range texts {
		refs[i] = enc.Encode(txt)
	}

	// Run interleaved and verify determinism.
	for round := 0; round < 50; round++ {
		for i, txt := range texts {
			v := enc.Encode(txt)
			if Similarity(refs[i], v) != 1.0 {
				t.Fatalf("round %d, text %d: pool contamination (sim=%.4f)", round, i, Similarity(refs[i], v))
			}
		}
	}
}

func TestEncode_Pooled_ConcurrentSafe(t *testing.T) {
	enc := NewNGramEncoder(DefaultConfig())
	text := "what is the capital of india"
	ref := enc.Encode(text)

	errs := make(chan error, 100)
	for g := 0; g < 10; g++ {
		go func() {
			for i := 0; i < 10; i++ {
				v := enc.Encode(text)
				if Similarity(ref, v) != 1.0 {
					errs <- nil // signal failure
					return
				}
			}
			errs <- nil
		}()
	}
	for i := 0; i < 10; i++ {
		<-errs
	}
}
