package cache

import (
	"fmt"
	"time"

	"xordb/hdc"
)

const snapshotVersion = 2

// EntrySnapshot is a serializable representation of one cache entry.
type EntrySnapshot struct {
	Key      string
	VecData  []uint64  // hdc.Vector raw words; dims stored at Snapshot level
	Value    any
	Ts       time.Time
	Deadline time.Time // zero = never expires
}

// Snapshot is a serializable point-in-time copy of the cache state.
// Use Cache.Snapshot() to create and Cache.LoadSnapshot() to restore.
type Snapshot struct {
	Version  int
	Dims     int
	Capacity int
	Entries  []EntrySnapshot // MRU order — index 0 is most recently used
}

// Snapshot returns a point-in-time serializable copy of the cache.
// Expired entries are skipped.
func (c *Cache) Snapshot() Snapshot {
	c.mu.Lock()
	defer c.mu.Unlock()

	now := time.Now()
	entries := make([]EntrySnapshot, 0, c.lru.Len())
	for elem := c.lru.Front(); elem != nil; elem = elem.Next() {
		e := elem.Value.(*entry)
		if c.isExpired(e, now) {
			continue
		}
		entries = append(entries, EntrySnapshot{
			Key:      e.key,
			VecData:  e.vec.Data(),
			Value:    e.value,
			Ts:       e.ts,
			Deadline: e.deadline,
		})
	}

	return Snapshot{
		Version:  snapshotVersion,
		Dims:     c.dims,
		Capacity: c.capacity,
		Entries:  entries,
	}
}

// LoadSnapshot merges a snapshot into the live cache.
// Entries that are already expired at load time are skipped.
// Existing keys are overwritten. Returns an error on version or dims mismatch.
func (c *Cache) LoadSnapshot(s Snapshot) error {
	if s.Version != snapshotVersion {
		return fmt.Errorf("cache: snapshot version %d unsupported (want %d)", s.Version, snapshotVersion)
	}
	if s.Dims != 0 && s.Dims != c.dims {
		return fmt.Errorf("cache: snapshot dims %d does not match cache dims %d", s.Dims, c.dims)
	}

	now := time.Now()
	c.mu.Lock()
	defer c.mu.Unlock()

	// Inject in reverse (LRU-first) so that the MRU entry ends up at the
	// front of the list after all inserts.
	for i := len(s.Entries) - 1; i >= 0; i-- {
		es := s.Entries[i]
		if !es.Deadline.IsZero() && now.After(es.Deadline) {
			continue // already expired
		}
		if len(es.VecData) != hdc.NumWords(c.dims) {
			return fmt.Errorf("cache: entry %q: VecData length %d != expected %d",
				es.Key, len(es.VecData), hdc.NumWords(c.dims))
		}
		c.injectLocked(es)
	}
	return nil
}

// injectLocked inserts an EntrySnapshot directly, bypassing the encoder.
// Must be called with c.mu held.
func (c *Cache) injectLocked(es EntrySnapshot) {
	// Overwrite if key already exists.
	if elem, ok := c.index[es.Key]; ok {
		c.removeLocked(elem)
	}
	if c.lru.Len() >= c.capacity {
		c.evictLocked()
	}
	e := &entry{
		key:      es.Key,
		vec:      hdc.FromWords(c.dims, es.VecData),
		value:    es.Value,
		ts:       es.Ts,
		deadline: es.Deadline,
	}
	c.index[es.Key] = c.lru.PushFront(e)
}
