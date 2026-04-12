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

	hdc "github.com/Amansingh-afk/hdc-go"

	"github.com/Amansingh-afk/xordb"
	"github.com/Amansingh-afk/xordb/embed"
)

// ── Go benchmarks (machine-readable) ─────────────────────────────────────────

func BenchmarkXorDB_NGram_Set(b *testing.B) {
	for i := 0; i < b.N; i++ {
		db := xordb.New(xordb.WithCapacity(1000))
		for _, qp := range Dataset {
			db.Set(qp.Cached, qp.Answer)
		}
	}
}

func BenchmarkXorDB_NGram_Lookup(b *testing.B) {
	db := xordb.New(xordb.WithCapacity(1000))
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

	var precision, recall, f1, fpr float64
	if tp+fp > 0 {
		precision = float64(tp) / float64(tp+fp) * 100
	}
	if tp+fn > 0 {
		recall = float64(tp) / float64(tp+fn) * 100
	}
	if precision+recall > 0 {
		f1 = 2 * precision * recall / (precision + recall)
	}
	if fp+tn > 0 {
		fpr = float64(fp) / float64(fp+tn) * 100
	}

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
	fmt.Printf(
		"║  Dataset:        %-39s ║\n",
		fmt.Sprintf(
			"%d queries (%d match, %d neg, %d hard-neg)",
			n,
			catTotal["match"],
			catTotal["neg"],
			catTotal["hard-neg"],
		),
	)
	fmt.Printf("║  Precision:      %-39s ║\n", fmt.Sprintf("%.1f%% (%d/%d hits correct)", precision, tp, tp+fp))
	fmt.Printf("║  Recall:         %-39s ║\n", fmt.Sprintf("%.1f%% (%d/%d matches found)", recall, tp, tp+fn))
	fmt.Printf("║  F1 Score:       %-39s ║\n", fmt.Sprintf("%.1f%%", f1))
	fmt.Printf("║  FP Rate:        %-39s ║\n", fmt.Sprintf("%.1f%% (%d/%d wrong hits)", fpr, fp, fp+tn))
	fmt.Printf("║  False neg:      %-39s ║\n", fmt.Sprintf("%d  (should hit, got miss)", fn))
	fmt.Printf("║  Total time:     %-39s ║\n", elapsed.Round(time.Microsecond))
	fmt.Printf("║  Avg latency:    %-39s ║\n", fmt.Sprintf("%v / query", avgLatency.Round(time.Microsecond)))
	fmt.Printf("║  Heap (Go):      %-39s ║\n", fmt.Sprintf("%.2f MB", float64(m.Alloc)/(1024*1024)))
	fmt.Printf("║  RSS (process):  %-39s ║\n", fmt.Sprintf("%.2f MB", rssMB))
	fmt.Printf("║  Dependencies:   %-39s ║\n", deps)
	fmt.Printf("║  Threshold:      %-39s ║\n", threshold)
	fmt.Println("╠══════════════════════════════════════════════════════════╣")

	// Category breakdown.
	for _, cat := range []string{"match", "neg", "hard-neg"} {
		if catTotal[cat] > 0 {
			fmt.Printf("║  %-12s    %-39s ║\n", cat+":", fmt.Sprintf("%d/%d correct", catCorrect[cat], catTotal[cat]))
		}
	}

	fmt.Println("╚══════════════════════════════════════════════════════════╝")
	fmt.Println()
}

// ── Round 1: N-gram HDC ──────────────────────────────────────────────────────

func TestXorDB_NGram_Report(t *testing.T) {
	db := xordb.New(
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

	printReport(t, "xordb — N-gram HDC Encoder", "0", "0.75 (default)", results, elapsed)
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

	printReport(t, "xordb — MiniLM Encoder (xordb/embed)", "onnxruntime_go + model file", "0.75 (default)", results, elapsed)
}

// ── MiniLM sweep: encode once, test thresholds on raw similarities ──────────

func TestXorDB_MiniLM_ThresholdSweep(t *testing.T) {
	if p := os.Getenv("ORT_LIB_PATH"); p != "" {
		ort.SetSharedLibraryPath(p)
	}

	enc, err := embed.NewMiniLMEncoder()
	if err != nil {
		t.Skipf("MiniLM encoder not available: %v", err)
	}
	defer enc.Close()

	// Encode all entries once
	type encoded struct {
		cachedVec hdc.Vector
		lookupVec hdc.Vector
		expectHit bool
		category  string
	}
	entries := make([]encoded, len(Dataset))
	for i, qp := range Dataset {
		entries[i] = encoded{
			cachedVec: enc.Encode(qp.Cached),
			lookupVec: enc.Encode(qp.Lookup),
			expectHit: qp.ExpectHit,
			category:  qp.Category,
		}
	}

	// Print similarity distribution
	var matchSims, nonMatchSims []float64
	for _, e := range entries {
		sim := hdc.Similarity(e.cachedVec, e.lookupVec)
		if e.expectHit {
			matchSims = append(matchSims, sim)
		} else {
			nonMatchSims = append(nonMatchSims, sim)
		}
	}

	fmt.Println()
	fmt.Println("── MiniLM Similarity Distribution ──────────────────────")
	fmt.Printf("%-12s %8s %8s\n", "sim range", "matches", "non-match")
	fmt.Println("──────────── ──────── ────────")
	buckets := []struct{ lo, hi float64 }{
		{0.40, 0.45}, {0.45, 0.50}, {0.50, 0.55}, {0.55, 0.60},
		{0.60, 0.65}, {0.65, 0.70}, {0.70, 0.75}, {0.75, 0.80},
		{0.80, 0.85}, {0.85, 0.90}, {0.90, 0.95}, {0.95, 1.01},
	}
	for _, b := range buckets {
		mc, nc := 0, 0
		for _, s := range matchSims {
			if s >= b.lo && s < b.hi {
				mc++
			}
		}
		for _, s := range nonMatchSims {
			if s >= b.lo && s < b.hi {
				nc++
			}
		}
		fmt.Printf("[%.2f, %.2f) %8d %8d\n", b.lo, b.hi, mc, nc)
	}
	fmt.Printf("\nMatch:     min=%.4f  max=%.4f  mean=%.4f  (n=%d)\n",
		minSlice(matchSims), maxSlice(matchSims), meanSlice(matchSims), len(matchSims))
	fmt.Printf("Non-match: min=%.4f  max=%.4f  mean=%.4f  (n=%d)\n",
		minSlice(nonMatchSims), maxSlice(nonMatchSims), meanSlice(nonMatchSims), len(nonMatchSims))

	// Sweep thresholds
	thresholds := []float64{0.55, 0.60, 0.65, 0.70, 0.72, 0.75, 0.78, 0.80, 0.82, 0.85}

	fmt.Println()
	fmt.Println("── MiniLM Threshold Sweep ──────────────────────────────")
	fmt.Printf("%-8s %6s %6s %6s %6s %6s %6s %6s\n",
		"thresh", "TP", "FP", "FN", "TN", "prec%", "rec%", "f1%")
	fmt.Println("──────── ────── ────── ────── ────── ────── ────── ──────")

	for _, th := range thresholds {
		var tp, fp, fn, tn int
		for _, e := range entries {
			sim := hdc.Similarity(e.cachedVec, e.lookupVec)
			hit := sim >= th
			switch {
			case e.expectHit && hit:
				tp++
			case !e.expectHit && !hit:
				tn++
			case !e.expectHit && hit:
				fp++
			case e.expectHit && !hit:
				fn++
			}
		}
		prec, rec, f1 := 0.0, 0.0, 0.0
		if tp+fp > 0 {
			prec = float64(tp) / float64(tp+fp) * 100
		}
		if tp+fn > 0 {
			rec = float64(tp) / float64(tp+fn) * 100
		}
		if prec+rec > 0 {
			f1 = 2 * prec * rec / (prec + rec)
		}
		fmt.Printf("%-8.2f %6d %6d %6d %6d %5.1f%% %5.1f%% %5.1f%%\n",
			th, tp, fp, fn, tn, prec, rec, f1)
	}
	fmt.Println()
}

func minSlice(s []float64) float64 {
	if len(s) == 0 {
		return 0
	}
	m := s[0]
	for _, v := range s[1:] {
		if v < m {
			m = v
		}
	}
	return m
}

func maxSlice(s []float64) float64 {
	if len(s) == 0 {
		return 0
	}
	m := s[0]
	for _, v := range s[1:] {
		if v > m {
			m = v
		}
	}
	return m
}

func meanSlice(s []float64) float64 {
	if len(s) == 0 {
		return 0
	}
	var sum float64
	for _, v := range s {
		sum += v
	}
	return sum / float64(len(s))
}
