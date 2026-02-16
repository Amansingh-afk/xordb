// Package hdc implements a Hyperdimensional Computing engine.
// Vectors are bitpacked []uint64 slices; all similarity operations are bitwise.
package hdc

import "math/bits"

// Vector is an immutable bitpacked hypervector.
// Padding bits in the final word are always zero.
type Vector struct {
	dims int
	data []uint64
}

// New returns a zero-valued Vector of the given dimension.
func New(dims int) Vector {
	if dims <= 0 {
		panic("hdc: dims must be positive")
	}
	return Vector{dims: dims, data: make([]uint64, numWords(dims))}
}

// FromWords constructs a Vector from a raw word slice.
// len(data) must equal ceil(dims/64). Padding bits are zeroed automatically.
func FromWords(dims int, data []uint64) Vector {
	if dims <= 0 {
		panic("hdc: dims must be positive")
	}
	needed := numWords(dims)
	if len(data) != needed {
		panic("hdc: data length does not match dims")
	}
	copied := make([]uint64, needed)
	copy(copied, data)
	zeroPadding(copied, dims)
	return Vector{dims: dims, data: copied}
}

func (v Vector) Dims() int { return v.dims }

// Clone returns an independent copy of v.
func (v Vector) Clone() Vector {
	data := make([]uint64, len(v.data))
	copy(data, v.data)
	return Vector{dims: v.dims, data: data}
}

// Permute performs a cyclic right-shift of the bit array by one position:
// result[i] = v[(i+1) % dims].
// Applying Permute dims times returns the original vector.
// Used for positional encoding in sequences.
func (v Vector) Permute() Vector {
	result := v.Clone()
	w := len(result.data)

	bit0 := result.data[0] & 1
	for i := 0; i < w-1; i++ {
		result.data[i] = (result.data[i] >> 1) | ((result.data[i+1] & 1) << 63)
	}
	highBit := uint((v.dims - 1) % 64)
	result.data[w-1] = (result.data[w-1] >> 1) | (bit0 << highBit)

	return result
}

// Bundle returns the majority-vote superposition of the given vectors.
// All vectors must have the same dimension.
// With an even count, ties resolve to 0.
func Bundle(vecs ...Vector) Vector {
	if len(vecs) == 0 {
		panic("hdc: Bundle requires at least one vector")
	}
	requireSameDims(vecs...)

	dims := vecs[0].dims
	threshold := len(vecs) / 2

	counts := make([]int32, dims)
	for _, v := range vecs {
		for w, word := range v.data {
			base := w * 64
			limit := 64
			if base+limit > dims {
				limit = dims - base
			}
			for b := 0; b < limit; b++ {
				counts[base+b] += int32(word >> uint(b) & 1)
			}
		}
	}

	result := New(dims)
	for i, c := range counts {
		if int(c) > threshold {
			result.data[i/64] |= 1 << uint(i%64)
		}
	}
	return result
}

// Bind associates two vectors via XOR. The operation is its own inverse:
// Bind(Bind(a, b), b) == a.
func Bind(a, b Vector) Vector {
	requireSameDims(a, b)
	result := New(a.dims)
	for i := range result.data {
		result.data[i] = a.data[i] ^ b.data[i]
	}
	return result
}

// Similarity returns the normalized Hamming similarity in [0.0, 1.0].
// 1.0 = identical, 0.0 = opposite, ~0.5 = unrelated random vectors.
func Similarity(a, b Vector) float64 {
	requireSameDims(a, b)
	var diff int
	for i := range a.data {
		diff += bits.OnesCount64(a.data[i] ^ b.data[i])
	}
	return 1.0 - float64(diff)/float64(a.dims)
}

func numWords(dims int) int {
	return (dims + 63) / 64
}

func zeroPadding(data []uint64, dims int) {
	if rem := dims % 64; rem != 0 {
		data[len(data)-1] &= (uint64(1) << uint(rem)) - 1
	}
}

func requireSameDims(vecs ...Vector) {
	d := vecs[0].dims
	for _, v := range vecs[1:] {
		if v.dims != d {
			panic("hdc: dimension mismatch")
		}
	}
}
