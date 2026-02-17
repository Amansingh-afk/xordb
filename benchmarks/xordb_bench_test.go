package benchmarks

import (
	"bufio"
	"fmt"
	"os"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"

	ort "github.com/yalue/onnxruntime_go"

	"xordb"
	"xordb/embed"
)

// ── Go benchmarks (machine-readable) ─────────────────────────────────────────

func BenchmarkXorDB_NGram_Set(b *testing.B) {
	for i := 0; i < b.N; i++ {
		db := xordb.New(xordb.WithThreshold(0.65), xordb.WithCapacity(1000))
		for _, qp := range Dataset {
			db.Set(qp.Cached, qp.Answer)
		}
	}
}

func BenchmarkXorDB_NGram_Lookup(b *testing.B) {
	db := xordb.New(xordb.WithThreshold(0.65), xordb.WithCapacity(1000))
	for _, qp := range Dataset {
		db.Set(qp.Cached, qp.Answer)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, qp := range Dataset {
			db.Get(qp.Lookup)
		}
	}
}

// ── Human-readable reports (used by Docker) ──────────────────────────────────

type queryResult struct {
	lookup    string
	category  string
	expectHit bool
	gotHit    bool
	sim       float64
}

func (r queryResult) correct() bool {
	return r.expectHit == r.gotHit
}

// readRSSMB reads the process resident set size from /proc/self/status.
// Returns 0 on any error (non-Linux platforms, permission issues, etc.).
func readRSSMB() float64 {
	f, err := os.Open("/proc/self/status")
	if err != nil {
		return 0
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "VmRSS:") {
			fields := strings.Fields(line)
			if len(fields) >= 2 {
				kb, err := strconv.ParseFloat(fields[1], 64)
				if err == nil {
					return kb / 1024.0
				}
			}
		}
	}
	return 0
}

// printReport prints a formatted benchmark report with accuracy metrics.
func printReport(t *testing.T, title string, deps string, threshold string, results []queryResult, elapsed time.Duration) {
	t.Helper()

	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	rssMB := readRSSMB()

	n := len(results)
	avgLatency := elapsed / time.Duration(n)

	// Count classification outcomes.
	var tp, tn, fp, fn int
	for _, r := range results {
		switch {
		case r.expectHit && r.gotHit:
			tp++
		case !r.expectHit && !r.gotHit:
			tn++
		case !r.expectHit && r.gotHit:
			fp++
		case r.expectHit && !r.gotHit:
			fn++
		}
	}

	accuracy := float64(tp+tn) / float64(n) * 100
	hits := tp + fp

	// Count by category.
	catTotal := map[string]int{}
	catCorrect := map[string]int{}
	for _, r := range results {
		catTotal[r.category]++
		if r.correct() {
			catCorrect[r.category]++
		}
	}

	fmt.Println()
	fmt.Println("╔══════════════════════════════════════════════════════════╗")
	fmt.Printf("║  %-55s ║\n", title)
	fmt.Println("╠══════════════════════════════════════════════════════════╣")
	fmt.Printf("║  Dataset:        %-39s ║\n", fmt.Sprintf("%d queries (40 match, 15 neg, 10 hard, 10 edge)", n))
	fmt.Printf("║  Accuracy:       %-39s ║\n", fmt.Sprintf("%.1f%% (%d/%d correct)", accuracy, tp+tn, n))
	fmt.Printf("║  True pos:       %-39s ║\n", fmt.Sprintf("%d  (correct hits)", tp))
	fmt.Printf("║  True neg:       %-39s ║\n", fmt.Sprintf("%d  (correct misses)", tn))
	fmt.Printf("║  False pos:      %-39s ║\n", fmt.Sprintf("%d  (should miss, got hit)", fp))
	fmt.Printf("║  False neg:      %-39s ║\n", fmt.Sprintf("%d  (should hit, got miss)", fn))
	fmt.Printf("║  Raw hits:       %-39s ║\n", fmt.Sprintf("%d / %d", hits, n))
	fmt.Printf("║  Total time:     %-39s ║\n", elapsed.Round(time.Microsecond))
	fmt.Printf("║  Avg latency:    %-39s ║\n", fmt.Sprintf("%v / query", avgLatency.Round(time.Microsecond)))
	fmt.Printf("║  Heap (Go):      %-39s ║\n", fmt.Sprintf("%.2f MB", float64(m.Alloc)/(1024*1024)))
	fmt.Printf("║  RSS (process):  %-39s ║\n", fmt.Sprintf("%.2f MB", rssMB))
	fmt.Printf("║  Dependencies:   %-39s ║\n", deps)
	fmt.Printf("║  Threshold:      %-39s ║\n", threshold)
	fmt.Println("╠══════════════════════════════════════════════════════════╣")

	// Category breakdown.
	for _, cat := range []string{"match", "neg", "hard-neg", "edge"} {
		if catTotal[cat] > 0 {
			fmt.Printf("║  %-12s    %-39s ║\n", cat+":", fmt.Sprintf("%d/%d correct", catCorrect[cat], catTotal[cat]))
		}
	}

	fmt.Println("╠══════════════════════════════════════════════════════════╣")
	fmt.Println("║  Per-query breakdown:                                    ║")
	fmt.Println("║  ✓ = correct, ✗ = wrong                                 ║")
	fmt.Println("╠══════════════════════════════════════════════════════════╣")

	for _, r := range results {
		status := "✓"
		if !r.correct() {
			status = "✗"
		}
		tag := "MISS"
		if r.gotHit {
			tag = "HIT "
		}
		lookup := r.lookup
		if len(lookup) > 33 {
			lookup = lookup[:30] + "..."
		}
		fmt.Printf("║ %s %s  %.2f  %-7s %-28s ║\n", status, tag, r.sim, "["+r.category+"]", lookup)
	}

	fmt.Println("╚══════════════════════════════════════════════════════════╝")
	fmt.Println()
}

// ── Round 1: N-gram HDC ──────────────────────────────────────────────────────

func TestXorDB_NGram_Report(t *testing.T) {
	db := xordb.New(
		xordb.WithThreshold(0.65),
		xordb.WithCapacity(1000),
	)

	for _, qp := range Dataset {
		db.Set(qp.Cached, qp.Answer)
	}

	results := make([]queryResult, 0, len(Dataset))
	start := time.Now()
	for _, qp := range Dataset {
		_, ok, sim := db.Get(qp.Lookup)
		results = append(results, queryResult{qp.Lookup, qp.Category, qp.ExpectHit, ok, sim})
	}
	elapsed := time.Since(start)

	printReport(t, "xordb — N-gram HDC Encoder", "0", "0.65", results, elapsed)
}

// ── Round 2: MiniLM (xordb/embed) ───────────────────────────────────────────

func TestXorDB_MiniLM_Report(t *testing.T) {
	// In Docker, the ONNX runtime .so has the "lib" prefix but yalue/onnxruntime_go
	// looks for "onnxruntime.so". Point it at the actual versioned library if the
	// symlink isn't enough.
	if p := os.Getenv("ORT_LIB_PATH"); p != "" {
		ort.SetSharedLibraryPath(p)
	}

	enc, err := embed.NewMiniLMEncoder()
	if err != nil {
		t.Skipf("MiniLM encoder not available: %v (run: xordb-model download)", err)
	}
	defer enc.Close()

	db := xordb.NewWithEncoder(enc,
		xordb.WithThreshold(0.75),
		xordb.WithCapacity(1000),
	)

	for _, qp := range Dataset {
		db.Set(qp.Cached, qp.Answer)
	}

	results := make([]queryResult, 0, len(Dataset))
	start := time.Now()
	for _, qp := range Dataset {
		_, ok, sim := db.Get(qp.Lookup)
		results = append(results, queryResult{qp.Lookup, qp.Category, qp.ExpectHit, ok, sim})
	}
	elapsed := time.Since(start)

	printReport(t, "xordb — MiniLM Encoder (xordb/embed)", "onnxruntime_go + model file", "0.75", results, elapsed)
}
