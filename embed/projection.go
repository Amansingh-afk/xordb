// Package embed provides semantic encoding using local ML models.
// MiniLMEncoder implements hdc.Encoder via a quantized MiniLM ONNX model.
// Float embeddings → binary hypervectors via random hyperplane LSH.
package embed

import (
	"math"
	"math/rand"

	"github.com/Amansingh-afk/xordb/hdc"
)

// Projector converts float32 embeddings to binary hdc.Vector via random
// hyperplane LSH. Each bit = sign(dot(embedding, hyperplane[i])).
type Projector struct {
	embDims    int
	binaryDims int
	planes     [][]float32 // [binaryDims][embDims]
}

// NewProjector creates a Projector. Deterministic for a given seed.
func NewProjector(embDims, binaryDims int, seed uint64) *Projector {
	if embDims <= 0 {
		panic("embed: embDims must be positive")
	}
	if binaryDims <= 0 {
		panic("embed: binaryDims must be positive")
	}

	rng := rand.New(rand.NewSource(int64(seed))) //nolint:gosec
	planes := make([][]float32, binaryDims)
	for i := range planes {
		plane := make([]float32, embDims)
		for j := range plane {
			plane[j] = float32(rng.NormFloat64())
		}
		// normalize to unit length
		var norm float64
		for _, v := range plane {
			norm += float64(v) * float64(v)
		}
		norm = math.Sqrt(norm)
		if norm > 0 {
			scale := float32(1.0 / norm)
			for j := range plane {
				plane[j] *= scale
			}
		}
		planes[i] = plane
	}

	return &Projector{
		embDims:    embDims,
		binaryDims: binaryDims,
		planes:     planes,
	}
}

// Project converts a float32 embedding to a binary hdc.Vector.
func (p *Projector) Project(embedding []float32) hdc.Vector {
	if len(embedding) != p.embDims {
		panic("embed: embedding length does not match projector embDims")
	}

	words := make([]uint64, hdc.NumWords(p.binaryDims))
	for i, plane := range p.planes {
		if dotProduct(embedding, plane) >= 0 {
			words[i/64] |= 1 << uint(i%64)
		}
	}

	return hdc.FromWords(p.binaryDims, words)
}

func dotProduct(a, b []float32) float32 {
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}
