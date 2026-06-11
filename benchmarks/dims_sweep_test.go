package benchmarks

import (
	"fmt"
	"os"
	"sort"
	"testing"

	hdc "github.com/Amansingh-afk/hdc-go"
	"github.com/Amansingh-afk/xordb/embed"
	ort "github.com/yalue/onnxruntime_go"
)

// TestDiagnose_DimsSweep: recall@K of the rerank window at decreasing binary
// dims. Projection, scan, and memory scale linearly with dims.
func TestDiagnose_DimsSweep(t *testing.T) {
	if p := os.Getenv("ORT_LIB_PATH"); p != "" {
		ort.SetSharedLibraryPath(p)
	}
	enc, err := embed.NewMiniLMEncoder()
	if err != nil {
		t.Skipf("MiniLM encoder not available: %v", err)
	}
	defer enc.Close()

	n := len(Dataset)
	cachedEmb := make([][]float32, n)
	lookupEmb := make([][]float32, n)
	for i, qp := range Dataset {
		if cachedEmb[i], err = enc.Embed(qp.Cached); err != nil {
			t.Fatal(err)
		}
		if lookupEmb[i], err = enc.Embed(qp.Lookup); err != nil {
			t.Fatal(err)
		}
	}

	fmt.Println("\n══ binary dims sweep: recall@K of paired entry (rerank window) ══")
	fmt.Printf("%-8s %9s %9s %9s %9s %12s\n", "dims", "rec@1", "rec@4", "rec@16", "rec@64", "vec bytes")

	for _, dims := range []int{256, 512, 1024, 2048, 4096, 10000} {
		proj := hdc.NewProjector(384, dims, 0xDB_CAFE)
		cb := make([]hdc.Vector, n)
		lb := make([]hdc.Vector, n)
		for i := range Dataset {
			cb[i] = proj.ProjectFloat(cachedEmb[i])
			lb[i] = proj.ProjectFloat(lookupEmb[i])
		}

		ks := []int{1, 4, 16, 64}
		hits := make([]int, len(ks))
		total := 0
		for i, qp := range Dataset {
			if !qp.ExpectHit {
				continue
			}
			total++
			type scored struct {
				idx int
				sim float64
			}
			all := make([]scored, n)
			for j := 0; j < n; j++ {
				all[j] = scored{j, hdc.Similarity(lb[i], cb[j])}
			}
			sort.Slice(all, func(a, b int) bool { return all[a].sim > all[b].sim })
			for r, s := range all {
				if s.idx == i {
					for ki, k := range ks {
						if r < k {
							hits[ki]++
						}
					}
					break
				}
			}
		}

		fmt.Printf("%-8d", dims)
		for ki := range ks {
			fmt.Printf(" %8.1f%%", 100*float64(hits[ki])/float64(total))
		}
		fmt.Printf(" %12d\n", dims/8)
	}
}
