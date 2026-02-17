//go:build integration

package embed_test

import (
	"os"
	"testing"

	"xordb"
	"xordb/embed"
	"xordb/hdc"
)

// These tests require:
//   - ONNX Runtime shared library installed
//   - MiniLM model downloaded (via xordb-model download)
//
// Run with: go test -tags integration -v ./...

func skipIfNoModel(t *testing.T) {
	t.Helper()
	if _, err := embed.DefaultModelPath(); err != nil {
		if p := os.Getenv("XORDB_MODEL_PATH"); p == "" {
			t.Skip("skipping: ONNX model not found (run xordb-model download)")
		}
	}
}

func TestXorDB_WithMiniLMEncoder_SemanticCache(t *testing.T) {
	skipIfNoModel(t)

	enc, err := embed.NewMiniLMEncoder()
	if err != nil {
		t.Fatalf("NewMiniLMEncoder: %v", err)
	}
	defer enc.Close()

	db := xordb.NewWithEncoder(enc,
		xordb.WithThreshold(0.70),
		xordb.WithCapacity(100),
	)

	// Populate cache with some entries.
	db.Set("what is the capital of india", "New Delhi")
	db.Set("who wrote ramayana", "Valmiki")
	db.Set("what is the speed of light", "299,792,458 m/s")
	db.Set("largest planet in the solar system", "Jupiter")

	tests := []struct {
		query   string
		wantHit bool
		wantVal string
		note    string
	}{
		{
			query:   "what is the capital of india",
			wantHit: true,
			wantVal: "New Delhi",
			note:    "exact match",
		},
		{
			query:   "capital city of india",
			wantHit: true,
			wantVal: "New Delhi",
			note:    "semantic paraphrase",
		},
		{
			query:   "india's capital",
			wantHit: true,
			wantVal: "New Delhi",
			note:    "short paraphrase",
		},
		{
			query:   "who is the author of ramayana",
			wantHit: true,
			wantVal: "Valmiki",
			note:    "author vs wrote",
		},
		{
			query:   "how fast does light travel",
			wantHit: true,
			wantVal: "299,792,458 m/s",
			note:    "speed of light paraphrase",
		},
		{
			query:   "biggest planet in our solar system",
			wantHit: true,
			wantVal: "Jupiter",
			note:    "biggest vs largest",
		},
		{
			query:   "how do you bake a chocolate cake",
			wantHit: false,
			note:    "completely unrelated",
		},
		{
			query:   "what programming language is Go",
			wantHit: false,
			note:    "unrelated topic",
		},
	}

	for _, tt := range tests {
		t.Run(tt.note, func(t *testing.T) {
			v, ok, sim := db.Get(tt.query)
			t.Logf("query=%q  hit=%v  sim=%.4f  val=%v", tt.query, ok, sim, v)

			if ok != tt.wantHit {
				t.Fatalf("query %q: want hit=%v, got hit=%v (sim=%.4f)", tt.query, tt.wantHit, ok, sim)
			}
			if tt.wantHit && v != tt.wantVal {
				t.Fatalf("query %q: want val=%q, got val=%v", tt.query, tt.wantVal, v)
			}
		})
	}
}

func TestXorDB_WithMiniLMEncoder_SimilarityOrdering(t *testing.T) {
	skipIfNoModel(t)

	enc, err := embed.NewMiniLMEncoder()
	if err != nil {
		t.Fatalf("NewMiniLMEncoder: %v", err)
	}
	defer enc.Close()

	// Encode two semantically different sentences.
	v1 := enc.Encode("the cat sat on the mat")
	v2 := enc.Encode("the dog sat on the rug")
	v3 := enc.Encode("quantum physics is complex")

	sim12 := hdc.Similarity(v1, v2)
	sim13 := hdc.Similarity(v1, v3)

	t.Logf("cat/dog similarity: %.4f", sim12)
	t.Logf("cat/quantum similarity: %.4f", sim13)

	// Cat/dog should be more similar than cat/quantum.
	if sim12 <= sim13 {
		t.Fatalf("semantically similar pair (%.4f) should score higher than unrelated (%.4f)", sim12, sim13)
	}
}

func TestXorDB_WithMiniLMEncoder_Stats(t *testing.T) {
	skipIfNoModel(t)

	enc, err := embed.NewMiniLMEncoder()
	if err != nil {
		t.Fatalf("NewMiniLMEncoder: %v", err)
	}
	defer enc.Close()

	db := xordb.NewWithEncoder(enc,
		xordb.WithThreshold(0.70),
		xordb.WithCapacity(100),
	)

	db.Set("hello world", "greeting")
	db.Get("hello world")   // hit
	db.Get("random stuff")  // miss

	s := db.Stats()
	if s.Entries != 1 {
		t.Fatalf("want 1 entry, got %d", s.Entries)
	}
	if s.Hits < 1 {
		t.Fatalf("want at least 1 hit, got %d", s.Hits)
	}
}
