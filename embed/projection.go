// Package embed provides semantic encoding for xordb using local ML models.
//
// The core component is [MiniLMEncoder], which implements the [xordb/hdc.Encoder]
// interface using a quantized MiniLM ONNX model for sentence embeddings.
// Float embeddings are projected to binary hypervectors via random hyperplane LSH
// for fast Hamming-distance similarity in xordb's HDC engine.
package embed

import (
	"math"
	"math/rand"

	"xordb/hdc"
)

// Projector converts dense float32 embeddings to binary hdc.Vector via random
// hyperplane Locality-Sensitive Hashing (LSH).
//
// Each output bit is the sign of the dot product between the input embedding
// and a random hyperplane. Deterministic for a given seed.
type Projector struct {
	embDims    int         // input embedding dimensionality (e.g. 384 for MiniLM)
	binaryDims int         // output binary vector dimensionality (e.g. 10000)
	planes     [][]float32 // [binaryDims][embDims] random hyperplanes
}

// NewProjector creates a Projector that maps embDims-dimensional float32 vectors
// to binaryDims-bit hdc.Vector values. The random hyperplanes are generated
// deterministically from seed.
//
// Panics if embDims or binaryDims is <= 0.
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
			// Standard normal via Box-Muller transform.
			plane[j] = float32(rng.NormFloat64())
		}
		// Normalize the plane to unit length for numerical stability.
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
// Each bit i is 1 if dot(embedding, planes[i]) >= 0, else 0.
//
// The input embedding must have length equal to embDims.
// Panics if the length does not match.
func (p *Projector) Project(embedding []float32) hdc.Vector {
	if len(embedding) != p.embDims {
		panic("embed: embedding length does not match projector embDims")
	}

	words := make([]uint64, hdc.NumWords(p.binaryDims))
	for i, plane := range p.planes {
		dot := dotProduct(embedding, plane)
		if dot >= 0 {
			words[i/64] |= 1 << uint(i%64)
		}
	}

	return hdc.FromWords(p.binaryDims, words)
}

// dotProduct computes the dot product of two float32 slices of equal length.
func dotProduct(a, b []float32) float32 {
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}
