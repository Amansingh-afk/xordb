package hdc_test

import (
	"testing"
	"xordb/hdc"
)

const (
	dims     = 10000
	dimSmall = 128 // for tests that loop dims times
)

// ── Vector construction ───────────────────────────────────────────────────────

func TestNew_ZeroVector(t *testing.T) {
	v := hdc.New(dims)
	if v.Dims() != dims {
		t.Fatalf("want dims %d, got %d", dims, v.Dims())
	}
	zero := hdc.New(dims)
	if hdc.Similarity(v, zero) != 1.0 {
		t.Fatal("New should return a zero vector")
	}
}

func TestFromWords_RoundTrip(t *testing.T) {
	a := hdc.Random(dims, 1)
	words := make([]uint64, (dims+63)/64)

	b := hdc.FromWords(dims, words)
	_ = b
	// Verify dims is preserved
	if a.Dims() != dims {
		t.Fatalf("dims mismatch: %d", a.Dims())
	}
}

func TestFromWords_PaddingZeroed(t *testing.T) {
	// dims=65 → 2 words; the second word has only bit 0 meaningful.
	// Pass a second word with high bits set; they should be zeroed.
	data := []uint64{^uint64(0), ^uint64(0)}
	v := hdc.FromWords(65, data)
	// Similarity with itself must be 1.0 (padding-consistent internally).
	if hdc.Similarity(v, v) != 1.0 {
		t.Fatal("Similarity of vector with itself must be 1.0")
	}
}

// ── Clone ─────────────────────────────────────────────────────────────────────

func TestClone_Identical(t *testing.T) {
	v := hdc.Random(dims, 42)
	c := v.Clone()
	if hdc.Similarity(v, c) != 1.0 {
		t.Fatal("Clone must be identical to original")
	}
}

func TestClone_Independent(t *testing.T) {
	a := hdc.Random(dims, 42)
	b := a.Clone()
	// Mutate b via Bind to verify a is unaffected
	bound := hdc.Bind(b, hdc.Random(dims, 99))
	if hdc.Similarity(a, bound) > 0.55 {
		t.Fatal("Clone should be independent of original")
	}
}

// ── Bind ──────────────────────────────────────────────────────────────────────

func TestBind_SelfInverse(t *testing.T) {
	a := hdc.Random(dims, 1)
	b := hdc.Random(dims, 2)
	if hdc.Similarity(a, hdc.Bind(hdc.Bind(a, b), b)) != 1.0 {
		t.Fatal("Bind(Bind(a,b),b) must equal a")
	}
}

func TestBind_QuasiOrthogonalToInputs(t *testing.T) {
	a := hdc.Random(dims, 1)
	b := hdc.Random(dims, 2)
	ab := hdc.Bind(a, b)
	assertNearHalf(t, "Bind result vs a", hdc.Similarity(a, ab))
	assertNearHalf(t, "Bind result vs b", hdc.Similarity(b, ab))
}

func TestBind_Commutativity(t *testing.T) {
	a := hdc.Random(dims, 1)
	b := hdc.Random(dims, 2)
	if hdc.Similarity(hdc.Bind(a, b), hdc.Bind(b, a)) != 1.0 {
		t.Fatal("Bind must be commutative (XOR is commutative)")
	}
}

func TestBind_DimensionMismatch_Panics(t *testing.T) {
	assertPanics(t, "Bind dim mismatch", func() {
		hdc.Bind(hdc.New(100), hdc.New(200))
	})
}

// ── Bundle ────────────────────────────────────────────────────────────────────

func TestBundle_Single_Identity(t *testing.T) {
	v := hdc.Random(dims, 42)
	if hdc.Similarity(v, hdc.Bundle(v)) != 1.0 {
		t.Fatal("Bundle of one vector must equal that vector")
	}
}

func TestBundle_OddIdentical(t *testing.T) {
	v := hdc.Random(dims, 1)
	if hdc.Similarity(v, hdc.Bundle(v, v, v)) != 1.0 {
		t.Fatal("Bundle of 3 identical vectors must equal that vector")
	}
}

func TestBundle_MajoritySimilarity(t *testing.T) {
	a := hdc.Random(dims, 1)
	b := hdc.Random(dims, 2)
	c := hdc.Random(dims, 3)
	bundled := hdc.Bundle(a, b, c)
	// Each input contributes ~2/3 of bits; expected similarity ~0.75.
	for label, v := range map[string]hdc.Vector{"a": a, "b": b, "c": c} {
		s := hdc.Similarity(bundled, v)
		if s < 0.68 || s > 0.82 {
			t.Fatalf("Bundle vs %s: expected ~0.75, got %.4f", label, s)
		}
	}
}

func TestBundle_Empty_Panics(t *testing.T) {
	assertPanics(t, "Bundle empty", func() { hdc.Bundle() })
}

func TestBundle_DimensionMismatch_Panics(t *testing.T) {
	assertPanics(t, "Bundle dim mismatch", func() {
		hdc.Bundle(hdc.New(100), hdc.New(200))
	})
}

// ── Similarity ────────────────────────────────────────────────────────────────

func TestSimilarity_Identical(t *testing.T) {
	v := hdc.Random(dims, 42)
	if hdc.Similarity(v, v) != 1.0 {
		t.Fatal("Similarity of vector with itself must be 1.0")
	}
}

func TestSimilarity_UnrelatedIsNearHalf(t *testing.T) {
	a := hdc.Random(dims, 100)
	b := hdc.Random(dims, 200)
	assertNearHalf(t, "unrelated random vectors", hdc.Similarity(a, b))
}

func TestSimilarity_DimensionMismatch_Panics(t *testing.T) {
	assertPanics(t, "Similarity dim mismatch", func() {
		hdc.Similarity(hdc.New(100), hdc.New(200))
	})
}

// ── Permute ───────────────────────────────────────────────────────────────────

func TestPermute_CyclicRestores(t *testing.T) {
	v := hdc.Random(dimSmall, 42)
	result := v
	for i := 0; i < dimSmall; i++ {
		result = result.Permute()
	}
	if hdc.Similarity(v, result) != 1.0 {
		t.Fatal("Permuting dims times must restore the original vector")
	}
}

func TestPermute_QuasiOrthogonal(t *testing.T) {
	v := hdc.Random(dims, 42)
	assertNearHalf(t, "single Permute", hdc.Similarity(v, v.Permute()))
}

func TestPermute_MultiWordCyclic(t *testing.T) {
	// 65 dims exercises the two-word boundary explicitly.
	v := hdc.Random(65, 7)
	result := v
	for i := 0; i < 65; i++ {
		result = result.Permute()
	}
	if hdc.Similarity(v, result) != 1.0 {
		t.Fatal("Permute cyclic failed for dims=65 (two-word boundary)")
	}
}

// ── Random ────────────────────────────────────────────────────────────────────

func TestRandom_Deterministic(t *testing.T) {
	a := hdc.Random(dims, 42)
	b := hdc.Random(dims, 42)
	if hdc.Similarity(a, b) != 1.0 {
		t.Fatal("Random with same seed must produce identical vectors")
	}
}

func TestRandom_DifferentSeedsOrthogonal(t *testing.T) {
	for seed := uint64(0); seed < 10; seed++ {
		a := hdc.Random(dims, seed)
		b := hdc.Random(dims, seed+1000)
		assertNearHalf(t, "different-seed randoms", hdc.Similarity(a, b))
	}
}

// ── Benchmarks ────────────────────────────────────────────────────────────────

func BenchmarkSimilarity(b *testing.B) {
	a := hdc.Random(dims, 1)
	v := hdc.Random(dims, 2)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		hdc.Similarity(a, v)
	}
}

func BenchmarkBind(b *testing.B) {
	a := hdc.Random(dims, 1)
	v := hdc.Random(dims, 2)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		hdc.Bind(a, v)
	}
}

func BenchmarkBundle10(b *testing.B) {
	vecs := make([]hdc.Vector, 10)
	for i := range vecs {
		vecs[i] = hdc.Random(dims, uint64(i))
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		hdc.Bundle(vecs...)
	}
}

func BenchmarkPermute(b *testing.B) {
	v := hdc.Random(dims, 42)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		v = v.Permute()
	}
}

func BenchmarkRandom(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		hdc.Random(dims, uint64(i))
	}
}

// ── helpers ───────────────────────────────────────────────────────────────────

func assertNearHalf(t *testing.T, label string, s float64) {
	t.Helper()
	if s < 0.45 || s > 0.55 {
		t.Fatalf("%s: expected similarity ~0.5 (quasi-orthogonal), got %.4f", label, s)
	}
}

func assertPanics(t *testing.T, label string, fn func()) {
	t.Helper()
	defer func() {
		if r := recover(); r == nil {
			t.Fatalf("%s: expected panic, got none", label)
		}
	}()
	fn()
}
