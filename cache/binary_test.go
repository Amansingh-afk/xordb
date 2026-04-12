package cache_test

import (
	"bytes"
	"encoding/binary"
	"testing"
	"time"

	"github.com/Amansingh-afk/xordb/cache"
	"github.com/Amansingh-afk/hdc-go"
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

func TestDecodeSnapshot_BadMagic(t *testing.T) {
	data := make([]byte, 32)
	copy(data[0:4], "NOPE")
	_, err := cache.DecodeSnapshot(bytes.NewReader(data), 10000)
	if err == nil {
		t.Fatal("expected error for bad magic")
	}
}

func TestDecodeSnapshot_BadVersion(t *testing.T) {
	var buf bytes.Buffer
	snap := cache.Snapshot{Version: 2, Dims: 1000, Capacity: 10}
	cache.EncodeSnapshot(&buf, snap)
	data := buf.Bytes()
	// Corrupt version field (bytes 4-5) to 99
	binary.LittleEndian.PutUint16(data[4:6], 99)
	_, err := cache.DecodeSnapshot(bytes.NewReader(data), 1000)
	if err == nil {
		t.Fatal("expected error for bad version")
	}
}

func TestDecodeSnapshot_DimsMismatch(t *testing.T) {
	var buf bytes.Buffer
	snap := cache.Snapshot{Version: 2, Dims: 1000, Capacity: 10}
	cache.EncodeSnapshot(&buf, snap)
	_, err := cache.DecodeSnapshot(bytes.NewReader(buf.Bytes()), 2000)
	if err == nil {
		t.Fatal("expected error for dims mismatch")
	}
}

func TestDecodeSnapshot_CRCCorruption(t *testing.T) {
	dims := 1000
	nw := hdc.NumWords(dims)
	snap := cache.Snapshot{
		Version:  2,
		Dims:     dims,
		Capacity: 10,
		Entries: []cache.EntrySnapshot{
			{
				Key:     "key",
				VecData: make([]uint64, nw),
				Value:   "val",
				Ts:      time.Now(),
			},
		},
	}

	var buf bytes.Buffer
	cache.EncodeSnapshot(&buf, snap)
	data := buf.Bytes()

	// Flip a byte in the payload (after 32-byte header)
	if len(data) > 33 {
		data[33] ^= 0xFF
	}

	_, err := cache.DecodeSnapshot(bytes.NewReader(data), dims)
	if err == nil {
		t.Fatal("expected CRC error after corruption")
	}
}

func TestBinary_EmptySnapshot(t *testing.T) {
	snap := cache.Snapshot{Version: 2, Dims: 10000, Capacity: 1024}
	var buf bytes.Buffer
	if err := cache.EncodeSnapshot(&buf, snap); err != nil {
		t.Fatal(err)
	}

	got, err := cache.DecodeSnapshot(&buf, 10000)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Entries) != 0 {
		t.Errorf("expected 0 entries, got %d", len(got.Entries))
	}
}

func TestBinary_JSONValueTypes(t *testing.T) {
	dims := 100
	nw := hdc.NumWords(dims)
	vec := make([]uint64, nw)

	tests := []struct {
		name  string
		value any
		check func(t *testing.T, got any)
	}{
		{"string", "hello", func(t *testing.T, got any) {
			if got != "hello" {
				t.Errorf("want hello, got %v", got)
			}
		}},
		{"int_as_float64", 42, func(t *testing.T, got any) {
			// JSON numbers decode as float64
			if got != float64(42) {
				t.Errorf("want 42, got %v (%T)", got, got)
			}
		}},
		{"nil", nil, func(t *testing.T, got any) {
			if got != nil {
				t.Errorf("want nil, got %v", got)
			}
		}},
		{"map", map[string]any{"a": float64(1)}, func(t *testing.T, got any) {
			m, ok := got.(map[string]any)
			if !ok {
				t.Fatalf("want map, got %T", got)
			}
			if m["a"] != float64(1) {
				t.Errorf("want a=1, got %v", m["a"])
			}
		}},
		{"slice", []any{"x", "y"}, func(t *testing.T, got any) {
			s, ok := got.([]any)
			if !ok {
				t.Fatalf("want slice, got %T", got)
			}
			if len(s) != 2 || s[0] != "x" {
				t.Errorf("unexpected slice: %v", s)
			}
		}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			snap := cache.Snapshot{
				Version:  2,
				Dims:     dims,
				Capacity: 10,
				Entries: []cache.EntrySnapshot{
					{Key: "k", VecData: vec, Value: tt.value, Ts: time.Now()},
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
			tt.check(t, got.Entries[0].Value)
		})
	}
}
