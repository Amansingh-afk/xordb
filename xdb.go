// Package xordb provides a semantic cache powered by Hyperdimensional Computing.
// Keys are encoded to hypervectors; Get returns the value stored under the
// most similar key above the configured threshold.
//
// Basic usage:
//
//	db := xordb.New()
//	db.Set("what is the capital of india", "Delhi")
//	v, ok, sim := db.Get("capital city of india") // semantic hit
package xordb

import (
	"xordb/cache"
	"xordb/hdc"
)

// Stats is a point-in-time snapshot of DB metrics.
type Stats struct {
	Entries     int
	Hits        uint64
	Misses      uint64
	Sets        uint64
	HitRate     float64
	AvgSimOnHit float64
}

// DB is a semantic cache. It is safe for concurrent use.
type DB struct {
	c *cache.Cache
}

// Option configures a DB.
type Option func(*dbOptions)

type dbOptions struct {
	dims             int
	threshold        float64
	capacity         int
	ngram            int
	seed             uint64
	stripPunctuation bool
}

func defaultOptions() dbOptions {
	return dbOptions{
		dims:      10000,
		threshold: 0.82,
		capacity:  1024,
		ngram:     3,
	}
}

// WithDims sets the hypervector dimension (default 10000).
// Higher values increase accuracy at the cost of memory and CPU.
func WithDims(n int) Option { return func(o *dbOptions) { o.dims = n } }

// WithThreshold sets the minimum similarity for a cache hit (default 0.82).
// Must be in (0, 1]. Raise to require closer matches; lower to be more permissive.
func WithThreshold(t float64) Option { return func(o *dbOptions) { o.threshold = t } }

// WithCapacity sets the maximum number of entries before LRU eviction (default 1024).
func WithCapacity(n int) Option { return func(o *dbOptions) { o.capacity = n } }

// WithNGramSize sets the character n-gram window size for encoding (default 3).
// Larger windows are more precise but less typo-tolerant.
func WithNGramSize(n int) Option { return func(o *dbOptions) { o.ngram = n } }

// WithSeed sets the encoder namespace seed (default 0).
// DBs with different seeds produce incompatible vectors.
func WithSeed(s uint64) Option { return func(o *dbOptions) { o.seed = s } }

// WithStripPunctuation enables punctuation removal during key normalization.
// Useful for natural-language queries; disable for code or structured keys.
func WithStripPunctuation(v bool) Option { return func(o *dbOptions) { o.stripPunctuation = v } }

// New creates a DB with the given options.
// Panics if any option value is invalid (e.g. Capacity=0, Threshold > 1).
func New(opts ...Option) *DB {
	o := defaultOptions()
	for _, opt := range opts {
		opt(&o)
	}
	enc := hdc.NewNGramEncoder(hdc.Config{
		Dims:             o.dims,
		NGramSize:        o.ngram,
		StripPunctuation: o.stripPunctuation,
		LongTextThresh:   200,
		ChunkSize:        128,
		Seed:             o.seed,
	})
	return &DB{c: cache.New(enc, cache.Options{
		Threshold: o.threshold,
		Capacity:  o.capacity,
	})}
}

// Set stores value under key. If the exact key already exists its value is
// updated and the entry is promoted to most-recently-used.
func (db *DB) Set(key string, value any) { db.c.Set(key, value) }

// Get returns the value stored under the most similar key at or above the threshold.
// Returns (value, true, similarity) on hit, or (nil, false, 0) on miss.
func (db *DB) Get(key string) (any, bool, float64) { return db.c.Get(key) }

// Delete removes the entry with the exact key string.
// Returns true if an entry was found and removed.
func (db *DB) Delete(key string) bool { return db.c.Delete(key) }

// Len returns the current number of cached entries.
func (db *DB) Len() int { return db.c.Len() }

// Stats returns a point-in-time snapshot of DB metrics.
func (db *DB) Stats() Stats {
	s := db.c.Stats()
	return Stats{
		Entries:     s.Entries,
		Hits:        s.Hits,
		Misses:      s.Misses,
		Sets:        s.Sets,
		HitRate:     s.HitRate,
		AvgSimOnHit: s.AvgSimOnHit,
	}
}
