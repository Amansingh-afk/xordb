package hdc

import "sync"

// bufPool recycles []uint64 word buffers and []int32 counts buffers to avoid
// per-Encode heap allocations. Both pools are keyed by the dims value so that
// a buffer obtained for one dimension is never accidentally reused for another.
//
// Zeroing happens on *get*, not put, so a stale buffer returned to the pool
// can never leak data into the next user.
type bufPool struct {
	words  sync.Pool // stores *[]uint64
	counts sync.Pool // stores *[]int32
	dims   int
}

func newBufPool(dims int) *bufPool {
	nw := numWords(dims)
	return &bufPool{
		dims: dims,
		words: sync.Pool{
			New: func() any {
				buf := make([]uint64, nw)
				return &buf
			},
		},
		counts: sync.Pool{
			New: func() any {
				buf := make([]int32, dims)
				return &buf
			},
		},
	}
}

// getWords returns a zeroed []uint64 slice of length numWords(dims).
func (p *bufPool) getWords() []uint64 {
	bp := p.words.Get().(*[]uint64)
	buf := *bp
	for i := range buf {
		buf[i] = 0
	}
	return buf
}

// putWords returns a word buffer to the pool.
func (p *bufPool) putWords(buf []uint64) {
	p.words.Put(&buf)
}

// getCounts returns a zeroed []int32 slice of length dims.
func (p *bufPool) getCounts() []int32 {
	bp := p.counts.Get().(*[]int32)
	buf := *bp
	for i := range buf {
		buf[i] = 0
	}
	return buf
}

// putCounts returns a counts buffer to the pool.
func (p *bufPool) putCounts(buf []int32) {
	p.counts.Put(&buf)
}
