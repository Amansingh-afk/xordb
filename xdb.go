// Package xordb — semantic cache powered by Hyperdimensional Computing.
//
//	db := xordb.New()
//	db.Set("what is the capital of india", "Delhi")
//	v, ok, sim := db.Get("capital city of india") // semantic hit
package xordb

import (
	"encoding/gob"
	"fmt"
	"os"
	"time"

	"xordb/cache"
	"xordb/hdc"
)

type Stats struct {
	Entries     int
	Hits        uint64
	Misses      uint64
	Sets        uint64
	Expired     uint64
	HitRate     float64
	AvgSimOnHit float64
}

// DB is a semantic cache. Safe for concurrent use.
type DB struct {
	c *cache.Cache
}

type Option func(*dbOptions)

type dbOptions struct {
	dims             int
	threshold        float64
	capacity         int
	ngram            int
	seed             uint64
	stripPunctuation bool
	ttl              time.Duration
}

func defaultOptions() dbOptions {
	return dbOptions{
		dims:      10000,
		threshold: 0.82,
		capacity:  1024,
		ngram:     3,
	}
}

func WithDims(n int) Option      { return func(o *dbOptions) { o.dims = n } }
func WithThreshold(t float64) Option { return func(o *dbOptions) { o.threshold = t } }
func WithCapacity(n int) Option  { return func(o *dbOptions) { o.capacity = n } }
func WithNGramSize(n int) Option { return func(o *dbOptions) { o.ngram = n } }
func WithSeed(s uint64) Option   { return func(o *dbOptions) { o.seed = s } }
func WithStripPunctuation(v bool) Option { return func(o *dbOptions) { o.stripPunctuation = v } }

// WithTTL sets the default TTL for cache entries. Zero = no expiry.
// Expired entries are lazily cleaned during Get scans.
func WithTTL(d time.Duration) Option { return func(o *dbOptions) { o.ttl = d } }

// New creates a DB with the built-in n-gram encoder.
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
		TTL:       o.ttl,
	})}
}

// NewWithEncoder — plug in any encoder (e.g. xordb/embed MiniLM).
// Encoding-related options (Dims, NGramSize, Seed etc.) are ignored since
// the encoder controls those.
func NewWithEncoder(enc hdc.Encoder, opts ...Option) *DB {
	if enc == nil {
		panic("xordb: encoder must not be nil")
	}
	o := defaultOptions()
	for _, opt := range opts {
		opt(&o)
	}
	return &DB{c: cache.New(enc, cache.Options{
		Threshold: o.threshold,
		Capacity:  o.capacity,
		TTL:       o.ttl,
	})}
}

func (db *DB) Set(key string, value any) { db.c.Set(key, value) }

// SetWithTTL — per-entry TTL that overrides the default. Zero = never expires.
func (db *DB) SetWithTTL(key string, value any, ttl time.Duration) {
	db.c.SetWithTTL(key, value, ttl)
}

// Get returns (value, true, similarity) on hit, (nil, false, 0) on miss.
func (db *DB) Get(key string) (any, bool, float64) { return db.c.Get(key) }

func (db *DB) Delete(key string) bool { return db.c.Delete(key) }
func (db *DB) Len() int              { return db.c.Len() }

// Save writes a snapshot of the cache to path using gob encoding.
// The write is atomic: data goes to path+".tmp" then renamed into place.
// Values of custom types must be registered first via RegisterGobType.
func (db *DB) Save(path string) error {
	snap := db.c.Snapshot()
	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return fmt.Errorf("xordb: save: %w", err)
	}
	if err := gob.NewEncoder(f).Encode(snap); err != nil {
		f.Close()
		os.Remove(tmp)
		return fmt.Errorf("xordb: save: encode: %w", err)
	}
	if err := f.Close(); err != nil {
		os.Remove(tmp)
		return fmt.Errorf("xordb: save: %w", err)
	}
	if err := os.Rename(tmp, path); err != nil {
		os.Remove(tmp)
		return fmt.Errorf("xordb: save: %w", err)
	}
	return nil
}

// Load merges a previously saved snapshot into the cache.
// Expired entries are skipped. Returns a wrapped os.ErrNotExist if the file is missing.
// Values of custom types must be registered first via RegisterGobType.
func (db *DB) Load(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("xordb: load: %w", err)
	}
	defer f.Close()
	var snap cache.Snapshot
	if err := gob.NewDecoder(f).Decode(&snap); err != nil {
		return fmt.Errorf("xordb: load: decode: %w", err)
	}
	if err := db.c.LoadSnapshot(snap); err != nil {
		return fmt.Errorf("xordb: load: %w", err)
	}
	return nil
}

// RegisterGobType registers a concrete type for gob encoding.
// Call once per custom type before Save or Load.
//
//	xordb.RegisterGobType(MyStruct{})
func RegisterGobType(v any) {
	gob.Register(v)
}

func (db *DB) Stats() Stats {
	s := db.c.Stats()
	return Stats{
		Entries:     s.Entries,
		Hits:        s.Hits,
		Misses:      s.Misses,
		Sets:        s.Sets,
		Expired:     s.Expired,
		HitRate:     s.HitRate,
		AvgSimOnHit: s.AvgSimOnHit,
	}
}
