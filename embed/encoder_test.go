package embed

import (
	"testing"
)

// ── unit tests (no ONNX model needed) ────────────────────────────────────────

func TestMeanPool(t *testing.T) {
	// 2 tokens, 3 dims: [[1,2,3], [3,4,5]]
	data := []float32{1, 2, 3, 3, 4, 5, 0, 0, 0} // 3rd token is padding
	result := meanPool(data, 2, 3, 3)

	want := []float32{2, 3, 4}
	for i, v := range result {
		if v != want[i] {
			t.Fatalf("meanPool[%d] = %f, want %f", i, v, want[i])
		}
	}
}

func TestMeanPool_SingleToken(t *testing.T) {
	data := []float32{1, 2, 3, 0, 0, 0}
	result := meanPool(data, 1, 2, 3)

	want := []float32{1, 2, 3}
	for i, v := range result {
		if v != want[i] {
			t.Fatalf("meanPool[%d] = %f, want %f", i, v, want[i])
		}
	}
}

func TestMeanPool_ZeroTokens(t *testing.T) {
	data := []float32{1, 2, 3}
	result := meanPool(data, 0, 1, 3)

	for i, v := range result {
		if v != 0 {
			t.Fatalf("meanPool[%d] = %f, want 0", i, v)
		}
	}
}

func TestL2Normalize(t *testing.T) {
	v := []float32{3, 4}
	l2Normalize(v)

	// 3/5 = 0.6, 4/5 = 0.8
	if abs32(v[0]-0.6) > 0.001 || abs32(v[1]-0.8) > 0.001 {
		t.Fatalf("l2Normalize([3,4]) = [%f,%f], want [0.6, 0.8]", v[0], v[1])
	}
}

func TestL2Normalize_ZeroVector(t *testing.T) {
	v := []float32{0, 0, 0}
	l2Normalize(v) // should not panic or produce NaN

	for i, x := range v {
		if x != 0 {
			t.Fatalf("l2Normalize zero vector[%d] = %f, want 0", i, x)
		}
	}
}

func TestCastInt32ToInt64(t *testing.T) {
	in := []int32{1, -2, 100, 0}
	out := castInt32ToInt64(in)

	if len(out) != len(in) {
		t.Fatalf("length mismatch: %d vs %d", len(out), len(in))
	}
	for i := range in {
		if out[i] != int64(in[i]) {
			t.Fatalf("castInt32ToInt64[%d] = %d, want %d", i, out[i], int64(in[i]))
		}
	}
}

func TestDefaultEncoderConfig(t *testing.T) {
	cfg := defaultEncoderConfig()
	if cfg.maxSeqLen != defaultMaxSeqLen {
		t.Fatalf("default maxSeqLen = %d, want %d", cfg.maxSeqLen, defaultMaxSeqLen)
	}
	if cfg.binaryDims != defaultBinaryDims {
		t.Fatalf("default binaryDims = %d, want %d", cfg.binaryDims, defaultBinaryDims)
	}
}

func TestModelDir_ReturnsNonEmpty(t *testing.T) {
	dir := ModelDir()
	if dir == "" {
		t.Fatal("ModelDir() returned empty string")
	}
}

func abs32(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}
