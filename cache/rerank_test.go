package cache

import (
	"bytes"
	"math"
	"testing"

	hdc "github.com/Amansingh-afk/hdc-go"
)

// fakeFloatEncoder maps known strings to fixed 8-dim embeddings and projects
// them to binary via a shared Projector. Unknown strings get a far-away
// embedding so they never match.
type fakeFloatEncoder struct {
	embs map[string][]float32
	proj *hdc.Projector
}

func newFakeFloatEncoder() *fakeFloatEncoder {
	return &fakeFloatEncoder{
		embs: map[string][]float32{
			"apple":  {1, 0.9, 0, 0, 0, 0, 0, 0},
			"apples": {0.9, 1, 0, 0, 0, 0, 0, 0},
			"banana": {0, 0, 1, 0.9, 0, 0, 0, 0},
			"rocket": {0, 0, 0, 0, 1, 0.9, 0, 0},
		},
		proj: hdc.NewProjector(8, 1024, 42),
	}
}

func (f *fakeFloatEncoder) emb(text string) []float32 {
	if e, ok := f.embs[text]; ok {
		return e
	}
	return []float32{0, 0, 0, 0, 0, 0, 0.9, 1}
}

func (f *fakeFloatEncoder) Encode(text string) hdc.Vector { return f.proj.ProjectFloat(f.emb(text)) }
func (f *fakeFloatEncoder) Embed(text string) ([]float32, error) {
	return f.emb(text), nil
}
func (f *fakeFloatEncoder) Project(emb []float32) hdc.Vector { return f.proj.ProjectFloat(emb) }

func TestRerank_CosineHitAndMiss(t *testing.T) {
	enc := newFakeFloatEncoder()
	c := New(enc, Options{Threshold: 0.9, Capacity: 16})

	if c.fenc == nil {
		t.Fatal("expected rerank to be enabled for FloatEncoder")
	}

	c.Set("apple", "fruit-a")
	c.Set("banana", "fruit-b")
	c.Set("rocket", "vehicle")

	v, ok, sim := c.Get("apples")
	if !ok || v != "fruit-a" {
		t.Fatalf("expected hit on fruit-a, got ok=%v v=%v sim=%.4f", ok, v, sim)
	}
	// cosine of {1,0.9,...} vs {0.9,1,...} = 1.8/1.81 ≈ 0.9945
	want := 1.8 / 1.81
	if math.Abs(sim-want) > 0.01 {
		t.Errorf("expected cosine ≈ %.4f, got %.4f", want, sim)
	}

	if _, ok, _ := c.Get("unrelated query"); ok {
		t.Error("expected miss for unrelated query")
	}
}

func TestRerank_DisabledByNegativeK(t *testing.T) {
	enc := newFakeFloatEncoder()
	c := New(enc, Options{Threshold: 0.6, Capacity: 16, RerankK: -1})
	if c.fenc != nil {
		t.Fatal("expected rerank disabled with RerankK=-1")
	}
}

func TestRerank_PersistRoundTrip(t *testing.T) {
	enc := newFakeFloatEncoder()
	c := New(enc, Options{Threshold: 0.9, Capacity: 16})
	c.Set("apple", "fruit-a")
	c.Set("banana", "fruit-b")

	var buf bytes.Buffer
	if err := EncodeSnapshot(&buf, c.Snapshot()); err != nil {
		t.Fatal(err)
	}
	snap, err := DecodeSnapshot(&buf, c.Dims())
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range snap.Entries {
		if e.Emb == nil {
			t.Fatalf("entry %q lost its embedding in round trip", e.Key)
		}
	}

	c2 := New(enc, Options{Threshold: 0.9, Capacity: 16})
	if err := c2.LoadSnapshot(snap); err != nil {
		t.Fatal(err)
	}
	v, ok, _ := c2.Get("apples")
	if !ok || v != "fruit-a" {
		t.Fatalf("rerank after reload: expected fruit-a hit, got ok=%v v=%v", ok, v)
	}
}

func TestRerank_LoadV2SnapshotReembeds(t *testing.T) {
	enc := newFakeFloatEncoder()
	c := New(enc, Options{Threshold: 0.9, Capacity: 16})
	c.Set("apple", "fruit-a")

	snap := c.Snapshot()
	snap.Version = 2
	for i := range snap.Entries {
		snap.Entries[i].Emb = nil // simulate pre-rerank snapshot
	}

	c2 := New(enc, Options{Threshold: 0.9, Capacity: 16})
	if err := c2.LoadSnapshot(snap); err != nil {
		t.Fatal(err)
	}
	v, ok, sim := c2.Get("apples")
	if !ok || v != "fruit-a" {
		t.Fatalf("expected re-embedded entry to hit, got ok=%v v=%v", ok, v)
	}
	if sim < 0.98 {
		t.Errorf("expected cosine score from re-embedded entry, got %.4f", sim)
	}
}
