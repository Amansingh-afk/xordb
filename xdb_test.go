package xordb_test

import (
	"fmt"
	"sync"
	"testing"

	"xordb"
	"xordb/hdc"
)

// ── construction ──────────────────────────────────────────────────────────────

func TestNew_Defaults(t *testing.T) {
	db := xordb.New()
	if db == nil {
		t.Fatal("New() must not return nil")
	}
	if db.Len() != 0 {
		t.Fatalf("fresh DB must be empty, got len=%d", db.Len())
	}
}

func TestNew_AllOptions(t *testing.T) {
	db := xordb.New(
		xordb.WithDims(512),
		xordb.WithThreshold(0.75),
		xordb.WithCapacity(64),
		xordb.WithNGramSize(4),
		xordb.WithSeed(42),
		xordb.WithStripPunctuation(true),
	)
	if db == nil {
		t.Fatal("New() with all options must not return nil")
	}
}

func TestNew_InvalidCapacity_Panics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for WithCapacity(0)")
		}
	}()
	xordb.New(xordb.WithCapacity(0))
}

func TestNew_InvalidThreshold_Panics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for WithThreshold(0)")
		}
	}()
	xordb.New(xordb.WithThreshold(0))
}

func TestNew_InvalidDims_Panics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for WithDims(0)")
		}
	}()
	xordb.New(xordb.WithDims(0))
}

func TestNew_InvalidNGramSize_Panics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for WithNGramSize(0)")
		}
	}()
	xordb.New(xordb.WithNGramSize(0))
}

// ── NewWithEncoder ────────────────────────────────────────────────────────────

func TestNewWithEncoder_CustomEncoder(t *testing.T) {
	enc := hdc.NewNGramEncoder(hdc.DefaultConfig())
	db := xordb.NewWithEncoder(enc)
	if db == nil {
		t.Fatal("NewWithEncoder must not return nil")
	}
	db.Set("hello", "world")
	v, ok, _ := db.Get("hello")
	if !ok || v != "world" {
		t.Fatalf("custom encoder DB must work, got ok=%v v=%v", ok, v)
	}
}

func TestNewWithEncoder_WithOptions(t *testing.T) {
	enc := hdc.NewNGramEncoder(hdc.DefaultConfig())
	db := xordb.NewWithEncoder(enc,
		xordb.WithThreshold(0.70),
		xordb.WithCapacity(64),
	)
	if db == nil {
		t.Fatal("NewWithEncoder with options must not return nil")
	}
}

func TestNewWithEncoder_NilEncoder_Panics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for nil encoder")
		}
	}()
	xordb.NewWithEncoder(nil)
}

// ── Set / Get ─────────────────────────────────────────────────────────────────

func TestDB_ExactHit(t *testing.T) {
	db := xordb.New()
	db.Set("what is the capital of india", "Delhi")

	v, ok, sim := db.Get("what is the capital of india")
	if !ok {
		t.Fatal("exact key must hit")
	}
	if v != "Delhi" {
		t.Fatalf("want Delhi, got %v", v)
	}
	if sim != 1.0 {
		t.Fatalf("exact hit must return sim=1.0, got %.4f", sim)
	}
}

func TestDB_Miss(t *testing.T) {
	db := xordb.New()
	db.Set("what is the capital of india", "Delhi")

	_, ok, sim := db.Get("how do you bake a chocolate cake")
	if ok {
		t.Fatal("unrelated query must miss at default threshold")
	}
	if sim != 0 {
		t.Fatalf("miss must return sim=0, got %.4f", sim)
	}
}

func TestDB_EmptyCache_Miss(t *testing.T) {
	db := xordb.New()
	_, ok, _ := db.Get("anything")
	if ok {
		t.Fatal("empty DB must always miss")
	}
}

func TestDB_SemanticHit(t *testing.T) {
	db := xordb.New(xordb.WithThreshold(0.65))
	db.Set("what is the capital of india", "Delhi")

	// Measured similarity: "capital city of india" → 0.7157 > 0.65
	v, ok, sim := db.Get("capital city of india")
	if !ok {
		t.Fatalf("expected semantic hit, got miss (sim would be ~0.72 > 0.65)")
	}
	if v != "Delhi" {
		t.Fatalf("want Delhi, got %v", v)
	}
	if sim < 0.65 {
		t.Fatalf("hit sim %.4f is below threshold", sim)
	}
}

func TestDB_BestMatch_Selected(t *testing.T) {
	db := xordb.New(xordb.WithThreshold(0.60))
	db.Set("what is the capital of india", "Delhi")
	db.Set("what is the capital of nepal", "Kathmandu")

	v, ok, _ := db.Get("capital city of nepal")
	if !ok {
		t.Fatal("expected hit")
	}
	if v != "Kathmandu" {
		t.Fatalf("best-match must return Kathmandu, got %v", v)
	}
}

func TestDB_Set_UpdateExactKey(t *testing.T) {
	db := xordb.New()
	db.Set("key", "first")
	db.Set("key", "second")

	v, ok, _ := db.Get("key")
	if !ok || v != "second" {
		t.Fatalf("want second after update, got %v ok=%v", v, ok)
	}
	if db.Len() != 1 {
		t.Fatalf("update must not create duplicate, len=%d", db.Len())
	}
}

func TestDB_AnyValueType(t *testing.T) {
	db := xordb.New()

	type payload struct{ n int }
	db.Set("struct", payload{42})
	db.Set("slice", []string{"a", "b"})
	db.Set("nil", nil)

	v, ok, _ := db.Get("struct")
	if !ok || v.(payload).n != 42 {
		t.Fatal("struct value roundtrip failed")
	}
	v, ok, _ = db.Get("nil")
	if !ok || v != nil {
		t.Fatal("nil value roundtrip failed")
	}
}

// ── Delete ────────────────────────────────────────────────────────────────────

func TestDB_Delete_Existing(t *testing.T) {
	db := xordb.New()
	db.Set("hello", "world")

	if !db.Delete("hello") {
		t.Fatal("Delete must return true for existing key")
	}
	_, ok, _ := db.Get("hello")
	if ok {
		t.Fatal("deleted entry must not be returned")
	}
}

func TestDB_Delete_Nonexistent(t *testing.T) {
	db := xordb.New()
	if db.Delete("ghost") {
		t.Fatal("Delete must return false for nonexistent key")
	}
}

// ── Len ───────────────────────────────────────────────────────────────────────

func TestDB_Len(t *testing.T) {
	db := xordb.New()
	for i := 0; i < 5; i++ {
		db.Set(fmt.Sprintf("entry-%d", i), i)
	}
	if db.Len() != 5 {
		t.Fatalf("want len=5, got %d", db.Len())
	}
}

// ── Stats ─────────────────────────────────────────────────────────────────────

func TestDB_Stats_Basic(t *testing.T) {
	db := xordb.New()
	db.Set("key", "value")
	db.Get("key")    // hit
	db.Get("other")  // miss
	db.Get("key")    // hit

	s := db.Stats()
	if s.Entries != 1 {
		t.Fatalf("want 1 entry, got %d", s.Entries)
	}
	if s.Hits != 2 {
		t.Fatalf("want 2 hits, got %d", s.Hits)
	}
	if s.Misses != 1 {
		t.Fatalf("want 1 miss, got %d", s.Misses)
	}
	wantRate := 2.0 / 3.0
	if diff := s.HitRate - wantRate; diff > 0.001 || diff < -0.001 {
		t.Fatalf("want hit rate %.4f, got %.4f", wantRate, s.HitRate)
	}
	if s.Sets != 1 {
		t.Fatalf("want 1 set, got %d", s.Sets)
	}
}

func TestDB_Stats_AvgSimOnHit(t *testing.T) {
	db := xordb.New()
	db.Set("hello", "world")
	db.Get("hello")
	db.Get("hello")

	s := db.Stats()
	if s.AvgSimOnHit != 1.0 {
		t.Fatalf("exact hits must give AvgSimOnHit=1.0, got %.4f", s.AvgSimOnHit)
	}
}

func TestDB_Stats_EmptyDB(t *testing.T) {
	s := xordb.New().Stats()
	if s.HitRate != 0 || s.AvgSimOnHit != 0 || s.Entries != 0 {
		t.Fatal("empty DB stats must be zero")
	}
}

// ── LRU via WithCapacity ──────────────────────────────────────────────────────

func TestDB_LRU_Eviction(t *testing.T) {
	db := xordb.New(xordb.WithCapacity(2))

	db.Set("alpha", 1)
	db.Set("beta", 2)
	db.Set("gamma", 3) // evicts alpha

	if db.Len() != 2 {
		t.Fatalf("capacity=2 but len=%d", db.Len())
	}
	_, ok, _ := db.Get("alpha")
	if ok {
		t.Fatal("alpha should have been evicted")
	}
}

// ── WithStripPunctuation ──────────────────────────────────────────────────────

func TestDB_WithStripPunctuation(t *testing.T) {
	db := xordb.New(xordb.WithStripPunctuation(true))
	db.Set("hello, world!", "greeting")

	v, ok, _ := db.Get("hello world")
	if !ok {
		t.Fatal("strip-punctuation DB must match clean query")
	}
	if v != "greeting" {
		t.Fatalf("want greeting, got %v", v)
	}
}

// ── WithSeed isolation ────────────────────────────────────────────────────────

func TestDB_WithSeed_Independent(t *testing.T) {
	db1 := xordb.New(xordb.WithSeed(1))
	db2 := xordb.New(xordb.WithSeed(2))

	db1.Set("hello", "from-db1")

	if db1.Len() != 1 {
		t.Fatal("db1 must have 1 entry")
	}
	if db2.Len() != 0 {
		t.Fatal("db2 must be independent of db1")
	}
}

// ── WithDims smoke test ───────────────────────────────────────────────────────

func TestDB_WithDims_ExactHit(t *testing.T) {
	db := xordb.New(xordb.WithDims(1000))
	db.Set("test key", 42)

	v, ok, _ := db.Get("test key")
	if !ok || v != 42 {
		t.Fatalf("WithDims(1000) must still produce exact hits, ok=%v v=%v", ok, v)
	}
}

// ── concurrency ───────────────────────────────────────────────────────────────

func TestDB_Concurrent(t *testing.T) {
	db := xordb.New(xordb.WithCapacity(128))

	var wg sync.WaitGroup
	for i := 0; i < 32; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			key := fmt.Sprintf("concurrent-key-%d", id%8)
			for j := 0; j < 20; j++ {
				if j%3 == 0 {
					db.Set(key, id*j)
				} else {
					db.Get(key)
				}
			}
		}(i)
	}
	wg.Wait()
}

// ── benchmarks ────────────────────────────────────────────────────────────────

func BenchmarkDB_Set(b *testing.B) {
	db := xordb.New(xordb.WithCapacity(b.N + 1))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		db.Set(fmt.Sprintf("benchmark key number %d", i), i)
	}
}

func BenchmarkDB_Get_100Entries(b *testing.B) {
	db := xordb.New(xordb.WithCapacity(1000))
	for i := 0; i < 100; i++ {
		db.Set(fmt.Sprintf("benchmark entry number %d", i), i)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		db.Get("benchmark entry number 50")
	}
}

func BenchmarkDB_Get_1000Entries(b *testing.B) {
	db := xordb.New(xordb.WithCapacity(2000))
	for i := 0; i < 1000; i++ {
		db.Set(fmt.Sprintf("benchmark entry number %d", i), i)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		db.Get("benchmark entry number 500")
	}
}

func BenchmarkDB_Get_10000Entries(b *testing.B) {
	db := xordb.New(xordb.WithCapacity(11000))
	for i := 0; i < 10000; i++ {
		db.Set(fmt.Sprintf("benchmark entry number %d", i), i)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		db.Get("benchmark entry number 5000")
	}
}
