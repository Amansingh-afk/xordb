//go:build integration

package embed

import (
	"os"
	"testing"

	"xordb/hdc"
)

// Integration tests require:
//   - ONNX Runtime shared library installed
//   - MiniLM model downloaded (via xordb-model download)
//
// Run with: go test -tags integration -v ./...

func skipIfNoModel(t *testing.T) {
	t.Helper()
	if _, err := DefaultModelPath(); err != nil {
		if p := os.Getenv("XORDB_MODEL_PATH"); p == "" {
			t.Skip("skipping: ONNX model not found (run xordb-model download)")
		}
	}
}

func TestMiniLMEncoder_Encode(t *testing.T) {
	skipIfNoModel(t)

	enc, err := NewMiniLMEncoder()
	if err != nil {
		t.Fatalf("NewMiniLMEncoder: %v", err)
	}
	defer enc.Close()

	v := enc.Encode("what is the capital of india")
	if v.Dims() != defaultBinaryDims {
		t.Fatalf("want dims=%d, got %d", defaultBinaryDims, v.Dims())
	}
}

func TestMiniLMEncoder_SemanticSimilarity(t *testing.T) {
	skipIfNoModel(t)

	enc, err := NewMiniLMEncoder()
	if err != nil {
		t.Fatalf("NewMiniLMEncoder: %v", err)
	}
	defer enc.Close()

	// Similar queries should have high similarity.
	v1 := enc.Encode("what is the capital of india")
	v2 := enc.Encode("capital city of india")
	simSimilar := hdc.Similarity(v1, v2)

	// Unrelated queries should have low similarity.
	v3 := enc.Encode("how to bake a chocolate cake")
	simUnrelated := hdc.Similarity(v1, v3)

	t.Logf("similar pair: %.4f", simSimilar)
	t.Logf("unrelated pair: %.4f", simUnrelated)

	if simSimilar < 0.70 {
		t.Fatalf("similar queries should have sim >= 0.70, got %.4f", simSimilar)
	}
	if simUnrelated > 0.60 {
		t.Fatalf("unrelated queries should have sim < 0.60, got %.4f", simUnrelated)
	}
	if simSimilar <= simUnrelated {
		t.Fatal("similar pair must score higher than unrelated pair")
	}
}

func TestMiniLMEncoder_Embed_RawVector(t *testing.T) {
	skipIfNoModel(t)

	enc, err := NewMiniLMEncoder()
	if err != nil {
		t.Fatalf("NewMiniLMEncoder: %v", err)
	}
	defer enc.Close()

	emb, err := enc.Embed("hello world")
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(emb) != miniLMEmbDims {
		t.Fatalf("want embedding dim=%d, got %d", miniLMEmbDims, len(emb))
	}

	// Check it's L2 normalized (norm should be ~1.0).
	var norm float64
	for _, v := range emb {
		norm += float64(v) * float64(v)
	}
	if abs64(norm-1.0) > 0.01 {
		t.Fatalf("embedding not L2 normalized: norm=%.6f", norm)
	}
}

func TestMiniLMEncoder_Deterministic(t *testing.T) {
	skipIfNoModel(t)

	enc, err := NewMiniLMEncoder()
	if err != nil {
		t.Fatalf("NewMiniLMEncoder: %v", err)
	}
	defer enc.Close()

	v1 := enc.Encode("test query")
	v2 := enc.Encode("test query")

	if hdc.Similarity(v1, v2) != 1.0 {
		t.Fatal("same input must produce identical vectors")
	}
}

func abs64(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
