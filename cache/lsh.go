package cache

import (
	"container/list"
	"math"
	"math/rand/v2"
)

// lshIndex implements Locality-Sensitive Hashing using bit-sampling for
// binary hypervectors with Hamming distance. Each of L tables uses k random
// bit positions to hash vectors into buckets.
type lshIndex struct {
	tables []lshTable
	hashes []lshHashFunc
	k, l   int
	dims   int
}

type lshTable struct {
	buckets map[uint64][]*list.Element
}

type lshHashFunc struct {
	bitPositions []int // k positions in [0, dims)
}

// newLSHIndex creates an LSH index with L tables, each using k random bit positions.
func newLSHIndex(dims, k, l int, seed uint64) *lshIndex {
	if k < 1 || k > 64 {
		panic("lsh: k must be in [1, 64]")
	}
	rng := rand.New(rand.NewChaCha8([32]byte(seedToBytes(seed))))

	hashes := make([]lshHashFunc, l)
	tables := make([]lshTable, l)
	for i := 0; i < l; i++ {
		positions := make([]int, k)
		for j := 0; j < k; j++ {
			positions[j] = rng.IntN(dims)
		}
		hashes[i] = lshHashFunc{bitPositions: positions}
		tables[i] = lshTable{buckets: make(map[uint64][]*list.Element)}
	}

	return &lshIndex{
		tables: tables,
		hashes: hashes,
		k:      k,
		l:      l,
		dims:   dims,
	}
}

// autoParams computes k and l from the similarity threshold using the formulas:
//
//	k = clamp(ceil(-3.0 / log2(threshold)), 6, 24)
//	l = clamp(ceil(log(0.1) / log(1 - threshold^k)), 8, 40)
func autoParams(threshold float64) (k, l int) {
	if threshold >= 1.0 {
		return 6, 8
	}
	kf := math.Ceil(-3.0 / math.Log2(threshold))
	k = clampInt(int(kf), 6, 24)

	pk := math.Pow(threshold, float64(k))
	lf := math.Ceil(math.Log(0.1) / math.Log(1.0-pk))
	l = clampInt(int(lf), 8, 40)
	return
}

// hashVec computes one hash key per table for the given raw vector data.
func (idx *lshIndex) hashVec(data []uint64) []uint64 {
	keys := make([]uint64, idx.l)
	for i, h := range idx.hashes {
		var key uint64
		for j, pos := range h.bitPositions {
			bit := (data[pos/64] >> uint(pos%64)) & 1
			key |= bit << uint(j)
		}
		keys[i] = key
	}
	return keys
}

// insert adds an element into all L tables using precomputed hash keys.
func (idx *lshIndex) insert(elem *list.Element, keys []uint64) {
	for i, key := range keys {
		idx.tables[i].buckets[key] = append(idx.tables[i].buckets[key], elem)
	}
}

// remove removes an element from all L tables using stored hash keys.
func (idx *lshIndex) remove(elem *list.Element, keys []uint64) {
	for i, key := range keys {
		bucket := idx.tables[i].buckets[key]
		for j, e := range bucket {
			if e == elem {
				// swap-remove
				bucket[j] = bucket[len(bucket)-1]
				bucket[len(bucket)-1] = nil
				bucket = bucket[:len(bucket)-1]
				if len(bucket) == 0 {
					delete(idx.tables[i].buckets, key)
				} else {
					idx.tables[i].buckets[key] = bucket
				}
				break
			}
		}
	}
}

// query returns deduplicated candidate elements from all L tables.
func (idx *lshIndex) query(keys []uint64) []*list.Element {
	seen := make(map[*list.Element]struct{})
	var candidates []*list.Element
	for i, key := range keys {
		for _, elem := range idx.tables[i].buckets[key] {
			if _, ok := seen[elem]; !ok {
				seen[elem] = struct{}{}
				candidates = append(candidates, elem)
			}
		}
	}
	return candidates
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func seedToBytes(seed uint64) []byte {
	b := make([]byte, 32)
	for i := 0; i < 8; i++ {
		b[i] = byte(seed >> (i * 8))
	}
	return b
}
