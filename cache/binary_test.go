package cache_test

import (
	"bytes"
	"testing"

	"xordb/cache"
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
