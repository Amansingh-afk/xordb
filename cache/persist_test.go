package cache_test

import (
	"testing"
	"time"

	"github.com/Amansingh-afk/xordb/cache"
	"github.com/Amansingh-afk/xordb/hdc"
)

func newTestCache(capacity int, threshold float64) *cache.Cache {
	enc := hdc.NewNGramEncoder(hdc.DefaultConfig())
	return cache.New(enc, cache.Options{Capacity: capacity, Threshold: threshold})
}

func TestSnapshot_RoundTrip(t *testing.T) {
	c := newTestCache(10, 0.99)
	c.Set("alpha", "A")
	c.Set("beta", "B")
	c.Set("gamma", "C")

	snap := c.Snapshot()
	if snap.Version != 2 {
		t.Fatalf("expected version 2 got %d", snap.Version)
	}
	if len(snap.Entries) != 3 {
		t.Fatalf("expected 3 entries got %d", len(snap.Entries))
	}

	c2 := newTestCache(10, 0.99)
	if err := c2.LoadSnapshot(snap); err != nil {
		t.Fatalf("LoadSnapshot: %v", err)
	}

	for _, key := range []string{"alpha", "beta", "gamma"} {
		v, ok, _ := c2.Get(key)
		if !ok {
			t.Errorf("expected hit for %q", key)
		}
		_ = v
	}
}

func TestSnapshot_MRUOrderPreserved(t *testing.T) {
	c := newTestCache(3, 0.99)
	c.Set("first", 1)
	c.Set("second", 2)
	c.Set("third", 3) // most recently used

	snap := c.Snapshot()
	// Entries[0] should be "third" (MRU)
	if snap.Entries[0].Key != "third" {
		t.Errorf("expected MRU at index 0, got %q", snap.Entries[0].Key)
	}
	if snap.Entries[2].Key != "first" {
		t.Errorf("expected LRU at index 2, got %q", snap.Entries[2].Key)
	}

	c2 := newTestCache(3, 0.99)
	if err := c2.LoadSnapshot(snap); err != nil {
		t.Fatal(err)
	}
	// Add one more to trigger eviction — "first" (LRU) should be evicted.
	c2.Set("fourth", 4)
	if _, ok, _ := c2.Get("first"); ok {
		t.Error("expected 'first' (LRU) to be evicted after load+Set")
	}
	if _, ok, _ := c2.Get("third"); !ok {
		t.Error("expected 'third' (MRU) to survive eviction")
	}
}

func TestSnapshot_TTL_ExpiredSkipped(t *testing.T) {
	c := newTestCache(10, 0.99)
	c.SetWithTTL("alive", "yes", time.Hour)
	c.SetWithTTL("dead", "no", time.Millisecond)
	time.Sleep(5 * time.Millisecond)

	snap := c.Snapshot()

	c2 := newTestCache(10, 0.99)
	if err := c2.LoadSnapshot(snap); err != nil {
		t.Fatal(err)
	}

	if _, ok, _ := c2.Get("alive"); !ok {
		t.Error("expected 'alive' to hit after load")
	}
	if _, ok, _ := c2.Get("dead"); ok {
		t.Error("expected 'dead' to be skipped (expired)")
	}
}

func TestSnapshot_TTL_DeadlinePreserved(t *testing.T) {
	c := newTestCache(10, 0.99)
	c.SetWithTTL("key", "val", time.Hour)

	snap := c.Snapshot()
	if snap.Entries[0].Deadline.IsZero() {
		t.Fatal("expected non-zero deadline in snapshot")
	}

	c2 := newTestCache(10, 0.99)
	if err := c2.LoadSnapshot(snap); err != nil {
		t.Fatal(err)
	}
	if _, ok, _ := c2.Get("key"); !ok {
		t.Error("expected hit after load")
	}
}

func TestSnapshot_EmptyCache(t *testing.T) {
	c := newTestCache(10, 0.99)
	snap := c.Snapshot()
	if len(snap.Entries) != 0 {
		t.Fatalf("expected 0 entries got %d", len(snap.Entries))
	}

	c2 := newTestCache(10, 0.99)
	if err := c2.LoadSnapshot(snap); err != nil {
		t.Fatalf("unexpected error loading empty snapshot: %v", err)
	}
	if c2.Len() != 0 {
		t.Errorf("expected empty cache after load, got %d", c2.Len())
	}
}

func TestSnapshot_VersionMismatch(t *testing.T) {
	c := newTestCache(10, 0.99)
	snap := c.Snapshot()
	snap.Version = 99

	c2 := newTestCache(10, 0.99)
	if err := c2.LoadSnapshot(snap); err == nil {
		t.Fatal("expected error on version mismatch")
	}
}

func TestSnapshot_DimsMismatch(t *testing.T) {
	enc1 := hdc.NewNGramEncoder(hdc.Config{Dims: 1000, NGramSize: 3, LongTextThresh: 200, ChunkSize: 128})
	c1 := cache.New(enc1, cache.Options{Capacity: 10, Threshold: 0.99})
	c1.Set("key", "val")
	snap := c1.Snapshot()

	enc2 := hdc.NewNGramEncoder(hdc.Config{Dims: 2000, NGramSize: 3, LongTextThresh: 200, ChunkSize: 128})
	c2 := cache.New(enc2, cache.Options{Capacity: 10, Threshold: 0.99})
	if err := c2.LoadSnapshot(snap); err == nil {
		t.Fatal("expected error on dims mismatch")
	}
}

func TestSnapshot_MergesIntoExistingCache(t *testing.T) {
	c1 := newTestCache(10, 0.99)
	c1.Set("from-snap", "snap-val")
	snap := c1.Snapshot()

	c2 := newTestCache(10, 0.99)
	c2.Set("pre-existing", "live-val")

	if err := c2.LoadSnapshot(snap); err != nil {
		t.Fatal(err)
	}

	if _, ok, _ := c2.Get("pre-existing"); !ok {
		t.Error("pre-existing entry should survive load")
	}
	if _, ok, _ := c2.Get("from-snap"); !ok {
		t.Error("snapshot entry should be present after load")
	}
}

func TestSnapshot_OverwritesDuplicateKey(t *testing.T) {
	c1 := newTestCache(10, 0.99)
	c1.Set("key", "from-snap")
	snap := c1.Snapshot()

	c2 := newTestCache(10, 0.99)
	c2.Set("key", "pre-existing")

	if err := c2.LoadSnapshot(snap); err != nil {
		t.Fatal(err)
	}

	v, ok, _ := c2.Get("key")
	if !ok {
		t.Fatal("expected hit")
	}
	if v != "from-snap" {
		t.Errorf("expected snapshot value to overwrite, got %v", v)
	}
}

func TestSnapshot_CapacityRespectedOnLoad(t *testing.T) {
	// Snapshot has 5 entries, target cache capacity is 3 → should only keep 3.
	c1 := newTestCache(10, 0.99)
	for i := 0; i < 5; i++ {
		c1.Set([]string{"a", "b", "c", "d", "e"}[i], i)
	}
	snap := c1.Snapshot()

	c2 := newTestCache(3, 0.99)
	if err := c2.LoadSnapshot(snap); err != nil {
		t.Fatal(err)
	}
	if c2.Len() > 3 {
		t.Errorf("expected at most 3 entries after load into capacity-3 cache, got %d", c2.Len())
	}
}
