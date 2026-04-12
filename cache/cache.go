// Package cache implements a thread-safe semantic cache backed by HDC vectors.
package cache

import (
	"container/list"
	"sync"
	"time"

	"github.com/Amansingh-afk/hdc-go"
)

type Options struct {
	Threshold float64       // minimum similarity for a hit
	Capacity  int           // max entries before LRU eviction
	TTL       time.Duration // default TTL; zero = no expiry

	LSHEnabled  *bool  // nil = auto (enabled if capacity >= 256)
	LSHK        int    // override auto-computed k; 0 = auto
	LSHL        int    // override auto-computed L; 0 = auto
	LSHFallback *bool  // nil or true = fallback to linear scan on LSH miss
	LSHSeed     uint64 // seed for LSH hash functions
}

func DefaultOptions() Options {
	return Options{Threshold: 0.82, Capacity: 1024}
}

type Stats struct {
	Entries       int
	Hits          uint64
	Misses        uint64
	Sets          uint64
	Expired       uint64
	HitRate       float64
	AvgSimOnHit   float64
	LSHCandidates uint64
	LSHFallbacks  uint64
}

type entry struct {
	key      string
	vec      hdc.Vector
	value    any
	ts       time.Time
	deadline time.Time // zero = never expires
	lshKeys  []uint64  // one per LSH table, nil if LSH disabled
}

// Cache — thread-safe semantic cache. Keys are encoded to hypervectors;
// Get returns the best match above threshold.
type Cache struct {
	mu        sync.Mutex
	enc       hdc.Encoder
	dims      int // vector dimensionality, used for snapshot validation
	lru       *list.List
	index     map[string]*list.Element
	threshold float64
	capacity  int
	ttl       time.Duration

	lsh         *lshIndex // nil if LSH disabled
	lshFallback bool      // fallback to linear scan on LSH miss

	hits          uint64
	misses        uint64
	sets          uint64
	expired       uint64
	simSum        float64
	lshCandidates uint64
	lshFallbacks  uint64
}

func New(enc hdc.Encoder, opts Options) *Cache {
	if opts.Capacity <= 0 {
		panic("cache: Options.Capacity must be positive")
	}
	if opts.Threshold <= 0 || opts.Threshold > 1 {
		panic("cache: Options.Threshold must be in (0, 1]")
	}

	dims := enc.Encode("").Dims()

	// LSH fallback defaults to true
	fallback := true
	if opts.LSHFallback != nil {
		fallback = *opts.LSHFallback
	}

	c := &Cache{
		enc:         enc,
		dims:        dims,
		lru:         list.New(),
		index:       make(map[string]*list.Element),
		threshold:   opts.Threshold,
		capacity:    opts.Capacity,
		ttl:         opts.TTL,
		lshFallback: fallback,
	}

	// Determine if LSH should be enabled
	lshEnabled := opts.LSHEnabled
	if lshEnabled == nil {
		auto := opts.Capacity >= 256
		lshEnabled = &auto
	}
	if *lshEnabled {
		k, l := opts.LSHK, opts.LSHL
		if k == 0 || l == 0 {
			ak, al := autoParams(opts.Threshold)
			if k == 0 {
				k = ak
			}
			if l == 0 {
				l = al
			}
		}
		c.lsh = newLSHIndex(dims, k, l, opts.LSHSeed)
	}

	return c
}

// Set stores value with the cache's default TTL.
func (c *Cache) Set(key string, value any) {
	c.setWithTTL(key, value, c.ttl)
}

// SetWithTTL — per-entry TTL override. Zero = never expires.
func (c *Cache) SetWithTTL(key string, value any, ttl time.Duration) {
	c.setWithTTL(key, value, ttl)
}

func (c *Cache) setWithTTL(key string, value any, ttl time.Duration) {
	if ttl < 0 {
		panic("cache: TTL must not be negative")
	}
	vec := c.enc.Encode(key)

	c.mu.Lock()
	defer c.mu.Unlock()

	c.sets++

	now := time.Now()
	dl := deadlineFrom(now, ttl)

	// update if exact key exists
	if elem, ok := c.index[key]; ok {
		e := elem.Value.(*entry)
		// Remove old LSH entries before updating vector
		if c.lsh != nil && e.lshKeys != nil {
			c.lsh.remove(elem, e.lshKeys)
		}
		e.value = value
		e.vec = vec
		e.ts = now
		e.deadline = dl
		if c.lsh != nil {
			e.lshKeys = c.lsh.hashVec(vec.RawData())
			c.lsh.insert(elem, e.lshKeys)
		}
		c.lru.MoveToFront(elem)
		return
	}

	if c.lru.Len() >= c.capacity {
		c.evictLocked()
	}

	e := &entry{key: key, vec: vec, value: value, ts: now, deadline: dl}
	if c.lsh != nil {
		e.lshKeys = c.lsh.hashVec(vec.RawData())
	}
	elem := c.lru.PushFront(e)
	c.index[key] = elem
	if c.lsh != nil {
		c.lsh.insert(elem, e.lshKeys)
	}
}

func deadlineFrom(now time.Time, ttl time.Duration) time.Time {
	if ttl > 0 {
		return now.Add(ttl)
	}
	return time.Time{}
}

// Get returns (value, true, similarity) on hit, (nil, false, 0) on miss.
func (c *Cache) Get(key string) (any, bool, float64) {
	vec := c.enc.Encode(key)

	c.mu.Lock()
	defer c.mu.Unlock()

	var bestElem *list.Element
	var bestSim float64

	if c.lsh != nil {
		keys := c.lsh.hashVec(vec.RawData())
		candidates := c.lsh.query(keys)
		c.lshCandidates += uint64(len(candidates))

		now := time.Now()
		for _, elem := range candidates {
			e := elem.Value.(*entry)
			if c.isExpired(e, now) {
				c.removeLocked(elem)
				c.expired++
				continue
			}
			if s := hdc.Similarity(vec, e.vec); s >= c.threshold && s > bestSim {
				bestSim = s
				bestElem = elem
			}
		}

		// Fallback to linear scan if LSH missed
		if bestElem == nil && c.lshFallback {
			c.lshFallbacks++
			bestElem, bestSim = c.scanLocked(vec)
		}
	} else {
		bestElem, bestSim = c.scanLocked(vec)
	}

	if bestElem == nil {
		c.misses++
		return nil, false, 0
	}

	c.lru.MoveToFront(bestElem)
	c.hits++
	c.simSum += bestSim
	return bestElem.Value.(*entry).value, true, bestSim
}

// Delete removes by exact key. Returns true if found.
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

// Dims returns the vector dimensionality.
func (c *Cache) Dims() int { return c.dims }

// Len returns the current number of cached entries.
func (c *Cache) Len() int {
	c.mu.Lock()
	n := c.lru.Len()
	c.mu.Unlock()
	return n
}

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
		Entries:       c.lru.Len(),
		Hits:          c.hits,
		Misses:        c.misses,
		Sets:          c.sets,
		Expired:       c.expired,
		HitRate:       hitRate,
		AvgSimOnHit:   avgSim,
		LSHCandidates: c.lshCandidates,
		LSHFallbacks:  c.lshFallbacks,
	}
}

// scanLocked — linear scan, returns best match above threshold.
// Expired entries lazily removed during scan (background goroutine nahi chahiye).
func (c *Cache) scanLocked(vec hdc.Vector) (*list.Element, float64) {
	var bestElem *list.Element
	var bestSim float64

	now := time.Now()
	for elem := c.lru.Front(); elem != nil; {
		e := elem.Value.(*entry)
		next := elem.Next()

		if c.isExpired(e, now) {
			c.removeLocked(elem)
			c.expired++
			elem = next
			continue
		}

		if s := hdc.Similarity(vec, e.vec); s >= c.threshold && s > bestSim {
			bestSim = s
			bestElem = elem
		}
		elem = next
	}
	return bestElem, bestSim
}

func (c *Cache) isExpired(e *entry, now time.Time) bool {
	return !e.deadline.IsZero() && now.After(e.deadline)
}

func (c *Cache) evictLocked() {
	if back := c.lru.Back(); back != nil {
		c.removeLocked(back)
	}
}

func (c *Cache) removeLocked(elem *list.Element) {
	e := elem.Value.(*entry)
	if c.lsh != nil && e.lshKeys != nil {
		c.lsh.remove(elem, e.lshKeys)
	}
	delete(c.index, e.key)
	c.lru.Remove(elem)
}
