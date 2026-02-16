// Package cache implements a thread-safe semantic cache backed by HDC vectors.
package cache

import (
	"container/list"
	"sync"
	"time"

	"xordb/hdc"
)

// Options configures a Cache.
type Options struct {
	Threshold float64 // minimum similarity for a hit (default 0.82)
	Capacity  int     // max entries before LRU eviction (default 1024)
}

// DefaultOptions returns production-ready defaults.
func DefaultOptions() Options {
	return Options{Threshold: 0.82, Capacity: 1024}
}

// Stats is a point-in-time snapshot of cache metrics.
type Stats struct {
	Entries     int
	Hits        uint64
	Misses      uint64
	Sets        uint64
	HitRate     float64
	AvgSimOnHit float64
}

type entry struct {
	key   string
	vec   hdc.Vector
	value any
	ts    time.Time
}

// Cache is a thread-safe semantic cache.
// Keys are encoded to hypervectors; Get returns the value stored under the
// most similar key above the configured threshold.
type Cache struct {
	mu        sync.Mutex
	enc       hdc.Encoder
	lru       *list.List
	index     map[string]*list.Element // exact-key → LRU element
	threshold float64
	capacity  int

	hits   uint64
	misses uint64
	sets   uint64
	simSum float64
}

// New creates a Cache using enc for key encoding.
// Panics if Capacity <= 0 or Threshold is outside (0, 1].
func New(enc hdc.Encoder, opts Options) *Cache {
	if opts.Capacity <= 0 {
		panic("cache: Options.Capacity must be positive")
	}
	if opts.Threshold <= 0 || opts.Threshold > 1 {
		panic("cache: Options.Threshold must be in (0, 1]")
	}
	return &Cache{
		enc:       enc,
		lru:       list.New(),
		index:     make(map[string]*list.Element),
		threshold: opts.Threshold,
		capacity:  opts.Capacity,
	}
}

// Set stores value under key.
// If the exact key already exists its value is updated in place and the entry
// is promoted to most-recently-used.
// If the cache is at capacity the least-recently-used entry is evicted first.
func (c *Cache) Set(key string, value any) {
	vec := c.enc.Encode(key) // encoding is lock-free

	c.mu.Lock()
	defer c.mu.Unlock()

	c.sets++

	if elem, ok := c.index[key]; ok {
		e := elem.Value.(*entry)
		e.value = value
		e.vec = vec
		e.ts = time.Now()
		c.lru.MoveToFront(elem)
		return
	}

	if c.lru.Len() >= c.capacity {
		c.evictLocked()
	}

	e := &entry{key: key, vec: vec, value: value, ts: time.Now()}
	c.index[key] = c.lru.PushFront(e)
}

// Get returns the value stored under the most similar key above the threshold.
// Returns (value, true, similarity) on a hit, or (nil, false, 0) on a miss.
// The matched entry is promoted to most-recently-used on a hit.
func (c *Cache) Get(key string) (any, bool, float64) {
	vec := c.enc.Encode(key) // lock-free

	c.mu.Lock()
	defer c.mu.Unlock()

	bestElem, bestSim := c.scanLocked(vec)
	if bestElem == nil {
		c.misses++
		return nil, false, 0
	}

	c.lru.MoveToFront(bestElem)
	c.hits++
	c.simSum += bestSim
	return bestElem.Value.(*entry).value, true, bestSim
}

// Delete removes the entry stored under the exact key string.
// The match is exact: the key must be byte-identical to the string passed to Set.
// Returns true if an entry was found and removed.
// To remove an entry whose key was normalised by the encoder, use the same
// original string that was passed to Set.
func (c *Cache) Delete(key string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()

	elem, ok := c.index[key]
	if !ok {
		return false
	}
	c.removeLocked(elem)
	return true
}

// Len returns the current number of cached entries.
func (c *Cache) Len() int {
	c.mu.Lock()
	n := c.lru.Len()
	c.mu.Unlock()
	return n
}

// Stats returns a point-in-time snapshot of cache metrics.
func (c *Cache) Stats() Stats {
	c.mu.Lock()
	defer c.mu.Unlock()

	total := c.hits + c.misses
	hitRate := 0.0
	if total > 0 {
		hitRate = float64(c.hits) / float64(total)
	}
	avgSim := 0.0
	if c.hits > 0 {
		avgSim = c.simSum / float64(c.hits)
	}

	return Stats{
		Entries:     c.lru.Len(),
		Hits:        c.hits,
		Misses:      c.misses,
		Sets:        c.sets,
		HitRate:     hitRate,
		AvgSimOnHit: avgSim,
	}
}

// scanLocked performs a linear similarity scan and returns the best-matching
// element at or above c.threshold, or nil if no match is found.
// Must be called with c.mu held.
func (c *Cache) scanLocked(vec hdc.Vector) (*list.Element, float64) {
	var bestElem *list.Element
	var bestSim float64

	for elem := c.lru.Front(); elem != nil; elem = elem.Next() {
		e := elem.Value.(*entry)
		if s := hdc.Similarity(vec, e.vec); s >= c.threshold && s > bestSim {
			bestSim = s
			bestElem = elem
		}
	}
	return bestElem, bestSim
}

func (c *Cache) evictLocked() {
	if back := c.lru.Back(); back != nil {
		c.removeLocked(back)
	}
}

func (c *Cache) removeLocked(elem *list.Element) {
	e := elem.Value.(*entry)
	delete(c.index, e.key)
	c.lru.Remove(elem)
}
