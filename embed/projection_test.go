package embed

import (
	"math"
	"testing"

	"xordb/hdc"
)

func TestNewProjector_Basic(t *testing.T) {
	p := NewProjector(384, 10000, 42)
	if p.embDims != 384 {
		t.Fatalf("want embDims=384, got %d", p.embDims)
	}
	if p.binaryDims != 10000 {
		t.Fatalf("want binaryDims=10000, got %d", p.binaryDims)
	}
	if len(p.planes) != 10000 {
		t.Fatalf("want 10000 planes, got %d", len(p.planes))
	}
	if len(p.planes[0]) != 384 {
		t.Fatalf("want plane dim=384, got %d", len(p.planes[0]))
	}
}

func TestNewProjector_InvalidDims_Panics(t *testing.T) {
	assertPanics(t, "embDims=0", func() { NewProjector(0, 100, 0) })
	assertPanics(t, "binaryDims=0", func() { NewProjector(100, 0, 0) })
}

func TestProjector_Deterministic(t *testing.T) {
	p1 := NewProjector(384, 10000, 42)
	p2 := NewProjector(384, 10000, 42)

	emb := makeTestEmbedding(384, 1)
	v1 := p1.Project(emb)
	v2 := p2.Project(emb)

	if hdc.Similarity(v1, v2) != 1.0 {
		t.Fatal("same seed must produce identical projections")
	}
}

func TestProjector_DifferentSeeds_QuasiOrthogonal(t *testing.T) {
	p1 := NewProjector(384, 10000, 1)
	p2 := NewProjector(384, 10000, 2)

	emb := makeTestEmbedding(384, 1)
	v1 := p1.Project(emb)
	v2 := p2.Project(emb)

	sim := hdc.Similarity(v1, v2)
	// Different random planes on the same embedding should produce ~0.5 similarity
	if sim < 0.40 || sim > 0.60 {
		t.Fatalf("different seeds should give ~0.5 similarity, got %.4f", sim)
	}
}

func TestProjector_SimilarEmbeddings_HighSimilarity(t *testing.T) {
	p := NewProjector(384, 10000, 42)

	emb1 := makeTestEmbedding(384, 1)
	emb2 := make([]float32, 384)
	copy(emb2, emb1)
	// Slightly perturb emb2
	for i := range emb2 {
		emb2[i] += 0.01
	}

	v1 := p.Project(emb1)
	v2 := p.Project(emb2)

	sim := hdc.Similarity(v1, v2)
	if sim < 0.85 {
		t.Fatalf("similar embeddings should project to similar vectors, got %.4f", sim)
	}
}

func TestProjector_OrthogonalEmbeddings_LowSimilarity(t *testing.T) {
	p := NewProjector(384, 10000, 42)

	emb1 := make([]float32, 384)
	emb2 := make([]float32, 384)
	// Create two orthogonal-ish embeddings
	for i := range emb1 {
		emb1[i] = float32(i) / 384.0
		emb2[i] = float32(384-i) / 384.0
	}

	v1 := p.Project(emb1)
	v2 := p.Project(emb2)

	sim := hdc.Similarity(v1, v2)
	// Not necessarily near 0.5 since these aren't truly orthogonal,
	// but should be clearly lower than identical
	if sim > 0.95 {
		t.Fatalf("different embeddings should not project to nearly identical vectors, got %.4f", sim)
	}
}

func TestProjector_WrongDims_Panics(t *testing.T) {
	p := NewProjector(384, 10000, 42)
	assertPanics(t, "wrong embedding dims", func() {
		p.Project(make([]float32, 100))
	})
}

func TestProjector_PlanesAreNormalized(t *testing.T) {
	p := NewProjector(384, 1000, 42)
	for i, plane := range p.planes {
		var norm float64
		for _, v := range plane {
			norm += float64(v) * float64(v)
		}
		norm = math.Sqrt(norm)
		if math.Abs(norm-1.0) > 0.001 {
			t.Fatalf("plane %d not normalized: norm=%.6f", i, norm)
		}
	}
}

func TestProjector_OutputDims(t *testing.T) {
	p := NewProjector(384, 10000, 42)
	emb := makeTestEmbedding(384, 1)
	v := p.Project(emb)
	if v.Dims() != 10000 {
		t.Fatalf("want output dims=10000, got %d", v.Dims())
	}
}

// ── benchmarks ────────────────────────────────────────────────────────────────

func BenchmarkProjector_Project(b *testing.B) {
	p := NewProjector(384, 10000, 42)
	emb := makeTestEmbedding(384, 1)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		p.Project(emb)
	}
}

func BenchmarkNewProjector(b *testing.B) {
	for i := 0; i < b.N; i++ {
		NewProjector(384, 10000, uint64(i))
	}
}

// ── helpers ───────────────────────────────────────────────────────────────────

func makeTestEmbedding(dims int, seed int64) []float32 {
	emb := make([]float32, dims)
	for i := range emb {
		emb[i] = float32(i+int(seed)) / float32(dims)
	}
	return emb
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
