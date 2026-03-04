package cache_test

import (
	"bytes"
	"testing"
	"time"

	"xordb/cache"
	"xordb/hdc"
)

func TestEncodeSnapshot_HeaderMagic(t *testing.T) {
	snap := cache.Snapshot{
		Version:  2,
		Dims:     10000,
		Capacity: 1024,
		Entries:  nil,
	}

	var buf bytes.Buffer
	if err := cache.EncodeSnapshot(&buf, snap); err != nil {
		t.Fatal(err)
	}

	data := buf.Bytes()
	if len(data) < 32 {
		t.Fatalf("expected at least 32 bytes header, got %d", len(data))
	}
	if string(data[:4]) != "XRDB" {
		t.Fatalf("expected magic XRDB, got %q", data[:4])
	}
}

func TestBinary_RoundTrip(t *testing.T) {
	dims := 1000
	nw := hdc.NumWords(dims)
	vec := make([]uint64, nw)
	for i := range vec {
		vec[i] = uint64(i + 1)
	}

	snap := cache.Snapshot{
		Version:  2,
		Dims:     dims,
		Capacity: 100,
		Entries: []cache.EntrySnapshot{
			{
				Key:      "hello",
				VecData:  vec,
				Value:    "world",
				Ts:       time.Now().Truncate(time.Microsecond),
				Deadline: time.Now().Add(time.Hour).Truncate(time.Microsecond),
			},
		},
	}

	var buf bytes.Buffer
	if err := cache.EncodeSnapshot(&buf, snap); err != nil {
		t.Fatal(err)
	}

	got, err := cache.DecodeSnapshot(&buf, dims)
	if err != nil {
		t.Fatal(err)
	}

	if got.Version != 2 {
		t.Errorf("version: want 2 got %d", got.Version)
	}
	if got.Dims != dims {
		t.Errorf("dims: want %d got %d", dims, got.Dims)
	}
	if got.Capacity != 100 {
		t.Errorf("capacity: want 100 got %d", got.Capacity)
	}
	if len(got.Entries) != 1 {
		t.Fatalf("entries: want 1 got %d", len(got.Entries))
	}
	e := got.Entries[0]
	if e.Key != "hello" {
		t.Errorf("key: want hello got %q", e.Key)
	}
	// JSON round-trips any → string
	if e.Value != "world" {
		t.Errorf("value: want world got %v", e.Value)
	}
}
