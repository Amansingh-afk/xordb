package cache_test

import (
	"fmt"
	"sync"
	"testing"

	"xordb/cache"
	"xordb/hdc"
)

// ── helpers ───────────────────────────────────────────────────────────────────

func newCache(threshold float64, capacity int) *cache.Cache {
	enc := hdc.NewNGramEncoder(hdc.DefaultConfig())
	return cache.New(enc, cache.Options{Threshold: threshold, Capacity: capacity})
}

// ── exact-match ───────────────────────────────────────────────────────────────

func TestCache_ExactHit(t *testing.T) {
	c := newCache(0.82, 16)
	c.Set("hello world", 42)

	v, ok, sim := c.Get("hello world")
	if !ok {
		t.Fatal("expected hit")
	}
	if v != 42 {
		t.Fatalf("want 42, got %v", v)
	}
	if sim != 1.0 {
		t.Fatalf("exact hit must return sim=1.0, got %.4f", sim)
	}
}

func TestCache_Miss(t *testing.T) {
	c := newCache(0.82, 16)
	c.Set("hello world", 42)

	_, ok, sim := c.Get("how do you bake a chocolate cake")
	if ok {
		t.Fatal("expected miss for unrelated query")
	}
	if sim != 0 {
		t.Fatalf("miss must return sim=0, got %.4f", sim)
	}
}

func TestCache_EmptyCache_Miss(t *testing.T) {
	c := newCache(0.82, 16)
	_, ok, _ := c.Get("anything")
	if ok {
		t.Fatal("empty cache must always miss")
	}
}

// ── semantic hit / miss calibrated to n-gram HDC actual scores ────────────────
// Measured similarity scores (DefaultConfig, dims=10000):
//   "what is the capital of india" vs "capital city of india"  → 0.7157
//   "what is the capital of india" vs "india's capital city"   → 0.6679
//   "what is the capital of india" vs "how do you bake a cake"  → ~0.52

func TestCache_SemanticHit_Paraphrase(t *testing.T) {
	c := newCache(0.65, 16)
	c.Set("what is the capital of india", "Delhi")

	v, ok, sim := c.Get("capital city of india")
	if !ok {
		t.Fatalf("expected semantic hit (sim=0.7157 > threshold=0.65), got miss")
	}
	if v != "Delhi" {
		t.Fatalf("want Delhi, got %v", v)
	}
	if sim < 0.65 {
		t.Fatalf("sim %.4f below threshold", sim)
	}
}

func TestCache_SemanticMiss_BelowThreshold(t *testing.T) {
	c := newCache(0.82, 16) // high threshold: only near-exact matches
	c.Set("what is the capital of india", "Delhi")

	// India paraphrase scores ~0.67, below 0.82 → miss
	_, ok, _ := c.Get("capital city of india")
	if ok {
		t.Fatal("paraphrase should miss at threshold=0.82")
	}
}

func TestCache_BestMatch_Selected(t *testing.T) {
	c := newCache(0.60, 16)

	c.Set("what is the capital of india", "Delhi")
	c.Set("what is the capital of nepal", "Kathmandu")

	// "capital of nepal" query: the Nepal entry is more similar
	v, ok, _ := c.Get("what is the capital of nepal")
	if !ok {
		t.Fatal("expected hit")
	}
	if v != "Kathmandu" {
		t.Fatalf("expected best-match to return Kathmandu, got %v", v)
	}
}

// ── update ────────────────────────────────────────────────────────────────────

func TestCache_Set_UpdateExactKey(t *testing.T) {
	c := newCache(0.82, 16)
	c.Set("key", "first")
	c.Set("key", "second")

	v, ok, _ := c.Get("key")
	if !ok {
		t.Fatal("expected hit after update")
	}
	if v != "second" {
		t.Fatalf("want second, got %v", v)
	}
	if c.Len() != 1 {
		t.Fatalf("update must not create a duplicate entry, len=%d", c.Len())
	}
}

// ── delete ────────────────────────────────────────────────────────────────────

func TestCache_Delete_Existing(t *testing.T) {
	c := newCache(0.82, 16)
	c.Set("key", "value")

	if !c.Delete("key") {
		t.Fatal("Delete must return true for existing key")
	}
	_, ok, _ := c.Get("key")
	if ok {
		t.Fatal("deleted entry must not be returned by Get")
	}
	if c.Len() != 0 {
		t.Fatalf("len must be 0 after delete, got %d", c.Len())
	}
}

func TestCache_Delete_Nonexistent(t *testing.T) {
	c := newCache(0.82, 16)
	if c.Delete("ghost") {
		t.Fatal("Delete must return false for nonexistent key")
	}
}

// ── LRU eviction ──────────────────────────────────────────────────────────────

func TestCache_LRU_EvictsOldest(t *testing.T) {
	// capacity=2; A then B inserted → A is LRU
	// inserting C must evict A
	// threshold=0.99 so only exact-key Gets hit (sim=1.0); unrelated entries score ~0.5
	c := newCache(0.99, 2)

	c.Set("alpha", 1)
	c.Set("beta", 2)
	c.Set("gamma", 3) // evicts alpha

	if c.Len() != 2 {
		t.Fatalf("capacity=2 but len=%d", c.Len())
	}

	_, ok, _ := c.Get("alpha")
	if ok {
		t.Fatal("alpha should have been evicted")
	}
}

func TestCache_LRU_AccessPromotes(t *testing.T) {
	// capacity=2; A then B inserted
	// accessing A promotes it → B becomes LRU
	// inserting C must evict B
	c := newCache(0.99, 2)

	c.Set("alpha", 1)
	c.Set("beta", 2)
	c.Get("alpha") // promote alpha
	c.Set("gamma", 3) // evicts beta (now LRU)

	_, aOk, _ := c.Get("alpha")
	_, bOk, _ := c.Get("beta")

	if !aOk {
		t.Fatal("alpha should still be cached (was promoted)")
	}
	if bOk {
		t.Fatal("beta should have been evicted")
	}
}

func TestCache_LRU_UpdatePromotes(t *testing.T) {
	c := newCache(0.99, 2)

	c.Set("alpha", 1)
	c.Set("beta", 2)
	c.Set("alpha", 99) // update promotes alpha → beta becomes LRU
	c.Set("gamma", 3)  // evicts beta

	_, bOk, _ := c.Get("beta")
	if bOk {
		t.Fatal("beta should have been evicted after alpha was updated")
	}
	v, aOk, _ := c.Get("alpha")
	if !aOk || v != 99 {
		t.Fatalf("alpha should be cached with updated value 99, got %v ok=%v", v, aOk)
	}
}

func TestCache_LRU_CapacityOne(t *testing.T) {
	c := newCache(0.99, 1)
	c.Set("a", 1)
	c.Set("b", 2)

	if c.Len() != 1 {
		t.Fatalf("capacity=1 but len=%d", c.Len())
	}
	_, aOk, _ := c.Get("a")
	v, bOk, _ := c.Get("b")
	if aOk {
		t.Fatal("a should be evicted")
	}
	if !bOk || v != 2 {
		t.Fatal("b should be cached")
	}
}

// ── Len ───────────────────────────────────────────────────────────────────────

func TestCache_Len(t *testing.T) {
	c := newCache(0.82, 10)
	for i := 0; i < 5; i++ {
		c.Set(fmt.Sprintf("key%d", i), i)
	}
	if c.Len() != 5 {
		t.Fatalf("want len=5, got %d", c.Len())
	}
}

// ── Stats ─────────────────────────────────────────────────────────────────────

func TestCache_Stats_HitRate(t *testing.T) {
	c := newCache(0.82, 16)
	c.Set("hello", "world")

	c.Get("hello") // hit
	c.Get("hello") // hit
	c.Get("zzzzz") // miss

	s := c.Stats()
	if s.Hits != 2 {
		t.Fatalf("want 2 hits, got %d", s.Hits)
	}
	if s.Misses != 1 {
		t.Fatalf("want 1 miss, got %d", s.Misses)
	}
	want := 2.0 / 3.0
	if diff := s.HitRate - want; diff > 0.001 || diff < -0.001 {
		t.Fatalf("want hit rate %.4f, got %.4f", want, s.HitRate)
	}
}

func TestCache_Stats_AvgSimOnHit(t *testing.T) {
	c := newCache(0.82, 16)
	c.Set("hello", "world")
	c.Get("hello") // sim=1.0
	c.Get("hello") // sim=1.0

	s := c.Stats()
	if s.AvgSimOnHit != 1.0 {
		t.Fatalf("exact hits must give AvgSimOnHit=1.0, got %.4f", s.AvgSimOnHit)
	}
}

func TestCache_Stats_Sets(t *testing.T) {
	c := newCache(0.82, 16)
	c.Set("a", 1)
	c.Set("b", 2)
	c.Set("a", 3) // update

	if s := c.Stats(); s.Sets != 3 {
		t.Fatalf("want 3 sets, got %d", s.Sets)
	}
}

func TestCache_Stats_Entries(t *testing.T) {
	c := newCache(0.82, 16)
	c.Set("a", 1)
	c.Set("b", 2)

	if s := c.Stats(); s.Entries != 2 {
		t.Fatalf("want 2 entries, got %d", s.Entries)
	}
}

func TestCache_Stats_NoHits_ZeroRates(t *testing.T) {
	c := newCache(0.82, 16)
	c.Get("nobody home") // miss only

	s := c.Stats()
	if s.HitRate != 0 {
		t.Fatalf("want HitRate=0 with no hits, got %.4f", s.HitRate)
	}
	if s.AvgSimOnHit != 0 {
		t.Fatalf("want AvgSimOnHit=0 with no hits, got %.4f", s.AvgSimOnHit)
	}
}

// ── concurrency ───────────────────────────────────────────────────────────────

func TestCache_Concurrent_SetGet(t *testing.T) {
	c := newCache(0.82, 128)

	const goroutines = 32
	const ops = 50

	var wg sync.WaitGroup
	wg.Add(goroutines)

	for g := 0; g < goroutines; g++ {
		go func(id int) {
			defer wg.Done()
			key := fmt.Sprintf("key-%d", id%8)
			for i := 0; i < ops; i++ {
				if i%3 == 0 {
					c.Set(key, id*i)
				} else {
					c.Get(key)
				}
			}
		}(g)
	}
	wg.Wait()
}

func TestCache_Concurrent_Delete(t *testing.T) {
	c := newCache(0.82, 128)
	for i := 0; i < 20; i++ {
		c.Set(fmt.Sprintf("key%d", i), i)
	}

	var wg sync.WaitGroup
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func(k int) {
			defer wg.Done()
			c.Delete(fmt.Sprintf("key%d", k))
		}(i)
	}
	wg.Wait()
}

// ── value types ───────────────────────────────────────────────────────────────

func TestCache_AnyValueType(t *testing.T) {
	c := newCache(0.82, 16)

	type payload struct{ msg string }

	c.Set("struct", payload{"hello"})
	c.Set("slice", []int{1, 2, 3})
	c.Set("nil", nil)

	v, ok, _ := c.Get("struct")
	if !ok || v.(payload).msg != "hello" {
		t.Fatal("struct value roundtrip failed")
	}
	v, ok, _ = c.Get("nil")
	if !ok || v != nil {
		t.Fatal("nil value roundtrip failed")
	}
}

// ── Options validation ────────────────────────────────────────────────────────

func TestNew_InvalidCapacity_Panics(t *testing.T) {
	enc := hdc.NewNGramEncoder(hdc.DefaultConfig())
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for Capacity=0")
		}
	}()
	cache.New(enc, cache.Options{Threshold: 0.82, Capacity: 0})
}

func TestNew_InvalidThresholdZero_Panics(t *testing.T) {
	enc := hdc.NewNGramEncoder(hdc.DefaultConfig())
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for Threshold=0")
		}
	}()
	cache.New(enc, cache.Options{Threshold: 0, Capacity: 16})
}

func TestNew_InvalidThresholdAboveOne_Panics(t *testing.T) {
	enc := hdc.NewNGramEncoder(hdc.DefaultConfig())
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for Threshold=1.1")
		}
	}()
	cache.New(enc, cache.Options{Threshold: 1.1, Capacity: 16})
}

// ── benchmarks ────────────────────────────────────────────────────────────────

func BenchmarkCache_Get_100Entries(b *testing.B) {
	c := newCache(0.82, 1000)
	for i := 0; i < 100; i++ {
		c.Set(fmt.Sprintf("entry number %d in the cache benchmark", i), i)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.Get("entry number 50 in the cache benchmark")
	}
}

func BenchmarkCache_Get_1000Entries(b *testing.B) {
	c := newCache(0.82, 2000)
	for i := 0; i < 1000; i++ {
		c.Set(fmt.Sprintf("entry number %d in the cache benchmark", i), i)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.Get("entry number 500 in the cache benchmark")
	}
}

func BenchmarkCache_Set(b *testing.B) {
	c := newCache(0.82, b.N+1)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.Set(fmt.Sprintf("key %d", i), i)
	}
}
