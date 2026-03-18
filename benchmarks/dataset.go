package benchmarks

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
)

// QueryPair represents a cached entry and a lookup query.
// ExpectHit indicates whether the lookup is semantically equivalent to the cached key.
type QueryPair struct {
	Cached    string `json:"cached"`
	Lookup    string `json:"lookup"`
	Answer    string `json:"answer"`
	ExpectHit bool   `json:"expect_hit"`
	Category  string `json:"category"`
}

// Dataset contains 100 realistic LLM query pairs for benchmarking.
// Loaded from data.json — the single source of truth shared with the Python benchmark.
var Dataset = mustLoadDataset()

func mustLoadDataset() []QueryPair {
	// Resolve data.json relative to this source file so it works
	// regardless of the working directory (go test, Docker, etc.).
	_, src, _, _ := runtime.Caller(0)
	path := filepath.Join(filepath.Dir(src), "data.json")

	data, err := os.ReadFile(path)
	if err != nil {
		panic("benchmarks: cannot read data.json: " + err.Error())
	}

	var pairs []QueryPair
	if err := json.Unmarshal(data, &pairs); err != nil {
		panic("benchmarks: cannot parse data.json: " + err.Error())
	}
	return pairs
}
