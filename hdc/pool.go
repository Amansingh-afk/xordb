package hdc

import "sync"

// bufPool recycles []uint64 and []int32 buffers via sync.Pool.
// GC pressure kam karne ke liye — zeroing happens on get, not put.
type bufPool struct {
	words  sync.Pool
	counts sync.Pool
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

func (p *bufPool) getWords() []uint64 {
	bp := p.words.Get().(*[]uint64)
	buf := *bp
	for i := range buf {
		buf[i] = 0
	}
	return buf
}

func (p *bufPool) putWords(buf []uint64) {
	p.words.Put(&buf)
}

func (p *bufPool) getCounts() []int32 {
	bp := p.counts.Get().(*[]int32)
	buf := *bp
	for i := range buf {
		buf[i] = 0
	}
	return buf
}

func (p *bufPool) putCounts(buf []int32) {
	p.counts.Put(&buf)
}
