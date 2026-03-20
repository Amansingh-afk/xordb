package cache

import (
	"container/list"
	"testing"

	"github.com/Amansingh-afk/xordb/hdc"
)

func TestLSH_HashDeterminism(t *testing.T) {
	dims := 1000
	idx := newLSHIndex(dims, 14, 20, 42)

	v := hdc.New(dims)
	data := v.RawData()

	keys1 := idx.hashVec(data)
	keys2 := idx.hashVec(data)

	for i := range keys1 {
		if keys1[i] != keys2[i] {
			t.Fatalf("hash not deterministic: table %d: %d != %d", i, keys1[i], keys2[i])
		}
	}
}

func TestLSH_InsertAndQuery(t *testing.T) {
	dims := 1000
	idx := newLSHIndex(dims, 6, 10, 42)

	ll := list.New()
	v := hdc.New(dims)
	elem := ll.PushFront(v)

	keys := idx.hashVec(v.RawData())
	idx.insert(elem, keys)

	candidates := idx.query(keys)
	if len(candidates) != 1 {
		t.Fatalf("expected 1 candidate, got %d", len(candidates))
	}
	if candidates[0] != elem {
		t.Fatal("returned wrong element")
	}
}

func TestLSH_Remove(t *testing.T) {
	dims := 1000
	idx := newLSHIndex(dims, 6, 10, 42)

	ll := list.New()
	v := hdc.New(dims)
	elem := ll.PushFront(v)

	keys := idx.hashVec(v.RawData())
	idx.insert(elem, keys)
	idx.remove(elem, keys)

	candidates := idx.query(keys)
	if len(candidates) != 0 {
		t.Fatalf("expected 0 candidates after remove, got %d", len(candidates))
	}
}

func TestLSH_EmptyIndex(t *testing.T) {
	dims := 1000
	idx := newLSHIndex(dims, 14, 20, 42)

	v := hdc.New(dims)
	keys := idx.hashVec(v.RawData())
	candidates := idx.query(keys)

	if len(candidates) != 0 {
		t.Fatalf("empty index must return 0 candidates, got %d", len(candidates))
	}
}

func TestLSH_QueryDeduplicates(t *testing.T) {
	dims := 1000
	idx := newLSHIndex(dims, 6, 20, 42) // low k, many tables → same elem in many buckets

	ll := list.New()
	v := hdc.New(dims)
	elem := ll.PushFront(v)

	keys := idx.hashVec(v.RawData())
	idx.insert(elem, keys)

	candidates := idx.query(keys)
	if len(candidates) != 1 {
		t.Fatalf("query must deduplicate: expected 1, got %d", len(candidates))
	}
}

func TestAutoParams_DefaultThreshold(t *testing.T) {
	k, l := autoParams(0.82)
	if k < 6 || k > 24 {
		t.Fatalf("k=%d out of [6,24]", k)
	}
	if l < 8 || l > 40 {
		t.Fatalf("l=%d out of [8,40]", l)
	}
}

func TestAutoParams_VariousThresholds(t *testing.T) {
	thresholds := []float64{0.70, 0.80, 0.85, 0.90, 0.95}
	for _, th := range thresholds {
		k, l := autoParams(th)
		if k < 6 || k > 24 {
			t.Errorf("threshold=%.2f: k=%d out of [6,24]", th, k)
		}
		if l < 8 || l > 40 {
			t.Errorf("threshold=%.2f: l=%d out of [8,40]", th, l)
		}
	}
}

func TestAutoParams_HighThreshold_LowK(t *testing.T) {
	k95, _ := autoParams(0.95)
	k70, _ := autoParams(0.70)
	// Higher threshold → fewer bits needed (vectors more similar → more bits match)
	// So k should be lower or equal for higher threshold
	// Actually: k = ceil(-3 / log2(threshold)). Higher threshold → log2 closer to 0 → larger k
	// Wait, log2(0.95) ≈ -0.074 → -3 / -0.074 ≈ 40.5 → clamped to 24
	// log2(0.70) ≈ -0.515 → -3 / -0.515 ≈ 5.8 → clamped to 6
	if k95 < k70 {
		t.Errorf("expected k95(%d) >= k70(%d)", k95, k70)
	}
}
