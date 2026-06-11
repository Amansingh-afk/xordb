package cache

import (
	"math"

	"github.com/Amansingh-afk/hdc-go"
)

// FloatEncoder is an Encoder that exposes the float embedding behind Encode
// (e.g. xordb/embed MiniLM). When the cache's encoder implements it, Get runs
// two-stage: Hamming top-K, then exact cosine on stored int8 embeddings.
// The hit threshold applies to the cosine score.
type FloatEncoder interface {
	hdc.Encoder
	Embed(text string) ([]float32, error)
	// Project converts an embedding from Embed into the binary vector that
	// Encode would produce for the same text.
	Project(emb []float32) hdc.Vector
}

const defaultRerankK = 16

// quantizeInt8 L2-normalizes emb and quantizes each component to int8.
func quantizeInt8(emb []float32) []int8 {
	var norm float64
	for _, v := range emb {
		norm += float64(v) * float64(v)
	}
	norm = math.Sqrt(norm)
	if norm == 0 {
		norm = 1
	}
	q := make([]int8, len(emb))
	for i, v := range emb {
		s := math.Round(float64(v) / norm * 127)
		if s > 127 {
			s = 127
		} else if s < -127 {
			s = -127
		}
		q[i] = int8(s)
	}
	return q
}

// cosineQ — cosine between a float query and an int8-quantized entry.
// Query side keeps full precision (asymmetric scoring).
func cosineQ(q []float32, d []int8) float64 {
	if len(q) != len(d) {
		return 0
	}
	var dot, nq, nd float64
	for i := range q {
		fq, fd := float64(q[i]), float64(d[i])
		dot += fq * fd
		nq += fq * fq
		nd += fd * fd
	}
	if nq == 0 || nd == 0 {
		return 0
	}
	return dot / (math.Sqrt(nq) * math.Sqrt(nd))
}
