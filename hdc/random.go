package hdc

import "math/rand"

// Random generates a deterministic pseudorandom Vector.
// Same (dims, seed) = same vector. Different seeds → quasi-orthogonal.
func Random(dims int, seed uint64) Vector {
	v := New(dims)
	r := rand.New(rand.NewSource(int64(seed))) //nolint:gosec
	for i := range v.data {
		v.data[i] = r.Uint64()
	}
	zeroPadding(v.data, dims)
	return v
}
