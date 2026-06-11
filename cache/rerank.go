package cache

import (
	"math"

	"github.com/Amansingh-afk/hdc-go"
)

// FloatEncoder is implemented by encoders that expose the raw float embedding
// behind Encode (e.g. xordb/embed MiniLMEncoder). When the cache's encoder
// implements it, Get runs a two-stage lookup: Hamming distance selects the
// top-K candidates, then exact cosine on stored int8-quantized embeddings
// picks the winner. The hit threshold applies to the cosine score.
//
// This recovers the recall/precision of the underlying float embeddings while
// keeping the binary scan speed: quantization compresses the gap between
// match and non-match scores, so a threshold in Hamming space cannot reach
// the high-precision operating points that cosine space offers.
type FloatEncoder interface {
	hdc.Encoder
	// Embed returns the raw float embedding for text.
	Embed(text string) ([]float32, error)
	// Project converts an embedding from Embed into the binary vector that
	// Encode would produce for the same text.
	Project(emb []float32) hdc.Vector
}

const defaultRerankK = 16

// quantizeInt8 L2-normalizes emb and quantizes each component to int8
// (scaled by 127). 384-dim embedding → 384 bytes per entry.
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

// cosineQ computes cosine similarity between a float query and an
// int8-quantized document embedding (asymmetric: the query keeps full
// precision, only the stored side is quantized).
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
