package hdc

import "math/bits"

// Vector is a bitpacked hypervector. Padding bits in the final word are always zero.
type Vector struct {
	dims int
	data []uint64
}

func New(dims int) Vector {
	if dims <= 0 {
		panic("hdc: dims must be positive")
	}
	return Vector{dims: dims, data: make([]uint64, numWords(dims))}
}

// FromWords constructs a Vector from raw words. len(data) must equal ceil(dims/64).
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

// Data returns a copy of the underlying words. Safe for serialization.
func (v Vector) Data() []uint64 {
	out := make([]uint64, len(v.data))
	copy(out, v.data)
	return out
}

func NumWords(dims int) int { return numWords(dims) }

// Bit returns the value (0 or 1) of the bit at position pos.
// Panics if pos is out of range [0, dims).
func (v Vector) Bit(pos int) uint64 {
	if pos < 0 || pos >= v.dims {
		panic("hdc: Bit position out of range")
	}
	return (v.data[pos/64] >> uint(pos%64)) & 1
}

// RawData returns a direct reference to the underlying words. Callers must not modify.
func (v Vector) RawData() []uint64 { return v.data }

func (v Vector) Clone() Vector {
	data := make([]uint64, len(v.data))
	copy(data, v.data)
	return Vector{dims: v.dims, data: data}
}

// Permute does a cyclic right-shift by 1 bit. Used for positional encoding.
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

// Bundle returns the majority-vote superposition. Ties resolve to 0.
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

// Bind = XOR. Self-inverse: Bind(Bind(a,b), b) == a.
func Bind(a, b Vector) Vector {
	requireSameDims(a, b)
	result := New(a.dims)
	for i := range result.data {
		result.data[i] = a.data[i] ^ b.data[i]
	}
	return result
}

// Similarity returns normalized Hamming similarity in [0, 1].
// 1 = identical, ~0.5 = random/unrelated, 0 = opposite.
func Similarity(a, b Vector) float64 {
	requireSameDims(a, b)
	var diff int
	for i := range a.data {
		diff += bits.OnesCount64(a.data[i] ^ b.data[i])
	}
	return 1.0 - float64(diff)/float64(a.dims)
}

// ── in-place ops (pooled encoder ke liye) ────────────────────────────────────

func permuteInto(dst, src Vector) {
	w := len(src.data)
	bit0 := src.data[0] & 1
	for i := 0; i < w-1; i++ {
		dst.data[i] = (src.data[i] >> 1) | ((src.data[i+1] & 1) << 63)
	}
	highBit := uint((src.dims - 1) % 64)
	dst.data[w-1] = (src.data[w-1] >> 1) | (bit0 << highBit)
}

func bindInto(dst, a, b Vector) {
	for i := range dst.data {
		dst.data[i] = a.data[i] ^ b.data[i]
	}
}

// bundleInto — majority vote into dst using counts as scratch space.
func bundleInto(dst Vector, counts []int32, vecs []Vector) {
	dims := dst.dims
	for i := 0; i < dims; i++ {
		counts[i] = 0
	}

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

	threshold := int32(len(vecs) / 2)
	for i := range dst.data {
		dst.data[i] = 0
	}
	for i := 0; i < dims; i++ {
		if counts[i] > threshold {
			dst.data[i/64] |= 1 << uint(i%64)
		}
	}
}

// vectorFromBuf wraps buf as a Vector without copying. Caller must not reuse buf.
func vectorFromBuf(dims int, buf []uint64) Vector {
	zeroPadding(buf, dims)
	return Vector{dims: dims, data: buf}
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
