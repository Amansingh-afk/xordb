package benchmarks

import (
	"fmt"
	"math"
	"os"
	"sort"
	"testing"

	hdc "github.com/Amansingh-afk/hdc-go"
	"github.com/Amansingh-afk/xordb"
	"github.com/Amansingh-afk/xordb/embed"
	ort "github.com/yalue/onnxruntime_go"
)

// TestDiagnose_RecallStages measures where recall dies in the MiniLM pipeline:
//
//	A. float cosine, brute force        → ceiling
//	B. binary projection, brute Hamming → quantization loss
//	C. LSH candidate pruning            → index loss
//	D. fixed threshold                  → cutoff loss
//
// Plus recall@K of the paired entry in binary space — predicts whether a
// two-stage rerank (Hamming top-K → exact cosine) recovers the ceiling.
func TestDiagnose_RecallStages(t *testing.T) {
	if p := os.Getenv("ORT_LIB_PATH"); p != "" {
		ort.SetSharedLibraryPath(p)
	}

	enc, err := embed.NewMiniLMEncoder()
	if err != nil {
		t.Skipf("MiniLM encoder not available: %v", err)
	}
	defer enc.Close()

	// ── Embed everything once (float32, L2-normalized) ──────────────────
	n := len(Dataset)
	cachedEmb := make([][]float32, n)
	lookupEmb := make([][]float32, n)
	for i, qp := range Dataset {
		if cachedEmb[i], err = enc.Embed(qp.Cached); err != nil {
			t.Fatalf("embed cached %d: %v", i, err)
		}
		if lookupEmb[i], err = enc.Embed(qp.Lookup); err != nil {
			t.Fatalf("embed lookup %d: %v", i, err)
		}
	}

	// ── Stage A: float cosine, brute force ──────────────────────────────
	fmt.Println("\n══ STAGE A: MiniLM float cosine (ceiling) ══")
	sweepStage(func(i int) (best float64, bestIdx int) {
		return bestMatch(n, func(j int) float64 { return cosine32(lookupEmb[i], cachedEmb[j]) })
	}, []float64{0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90})

	// ── Stage B: 1-bit sign projection, brute Hamming ────────────────────
	// Same projector params as embed.NewMiniLMEncoder defaults.
	const projSeed = 0xDB_CAFE
	proj1 := hdc.NewProjector(384, 10_000, projSeed)
	cachedBin := make([]hdc.Vector, n)
	lookupBin := make([]hdc.Vector, n)
	for i := range Dataset {
		cachedBin[i] = proj1.ProjectFloat(cachedEmb[i])
		lookupBin[i] = proj1.ProjectFloat(lookupEmb[i])
	}

	fmt.Println("\n══ STAGE B: 1-bit projection, brute-force Hamming ══")
	sweepStage(func(i int) (float64, int) {
		return bestMatch(n, func(j int) float64 { return hdc.Similarity(lookupBin[i], cachedBin[j]) })
	}, []float64{0.55, 0.58, 0.60, 0.62, 0.65, 0.70, 0.75})

	// Score distributions: how much does binarization compress the gap?
	printDistributions(n, cachedEmb, lookupEmb, cachedBin, lookupBin)

	// recall@K of the paired entry — the two-stage rerank predictor.
	fmt.Println("\n── recall@K (paired cached entry in binary top-K) ──")
	printRecallAtK(n, lookupBin, cachedBin)

	// ── Stage B variants: multi-bit thermometer encoding ──────────────────
	for _, bits := range []int{2, 4} {
		pm := hdc.NewProjectorMultiBit(384, 10_000, bits, projSeed)
		cb := make([]hdc.Vector, n)
		lb := make([]hdc.Vector, n)
		for i := range Dataset {
			cb[i] = pm.ProjectFloat(cachedEmb[i])
			lb[i] = pm.ProjectFloat(lookupEmb[i])
		}
		fmt.Printf("\n══ STAGE B': %d-bit thermometer projection ══\n", bits)
		sweepStage(func(i int) (float64, int) {
			return bestMatch(n, func(j int) float64 { return hdc.Similarity(lb[i], cb[j]) })
		}, []float64{0.55, 0.58, 0.60, 0.62, 0.65, 0.70, 0.75})
		fmt.Printf("── recall@K (%d-bit) ──\n", bits)
		printRecallAtK(n, lb, cb)
	}

	// ── Stage C: LSH pruning loss (fallback off vs on, threshold 0.75) ───
	fmt.Println("\n══ STAGE C: LSH candidate pruning ══")
	for _, fallback := range []bool{false, true} {
		db := xordb.NewWithEncoder(enc,
			xordb.WithCapacity(1000),
			xordb.WithLSH(true),
			xordb.WithLSHFallback(fallback),
		)
		for _, qp := range Dataset {
			db.Set(qp.Cached, qp.Answer)
		}
		var tp, fp, fn, tn int
		for _, qp := range Dataset {
			_, ok, _ := db.Get(qp.Lookup)
			switch {
			case qp.ExpectHit && ok:
				tp++
			case !qp.ExpectHit && !ok:
				tn++
			case !qp.ExpectHit && ok:
				fp++
			default:
				fn++
			}
		}
		prec, rec, f1 := metrics(tp, fp, fn)
		fmt.Printf("LSH fallback=%-5v  TP=%-4d FP=%-4d FN=%-4d TN=%-4d  prec=%5.1f%% rec=%5.1f%% f1=%5.1f%%\n",
			fallback, tp, fp, fn, tn, prec, rec, f1)
	}
	fmt.Println("\n(Stage D = stage B table at threshold 0.75 — the current default.)")
}

// TestDiagnose_Rerank measures the full xordb pipeline with two-stage cosine
// rerank enabled (encoder implements cache.FloatEncoder). Sweeps the hit
// threshold in cosine space and reports both hit-rate metrics and answer
// correctness (hit returned the paired entry's value, not a lookalike's).
func TestDiagnose_Rerank(t *testing.T) {
	if p := os.Getenv("ORT_LIB_PATH"); p != "" {
		ort.SetSharedLibraryPath(p)
	}

	enc, err := embed.NewMiniLMEncoder()
	if err != nil {
		t.Skipf("MiniLM encoder not available: %v", err)
	}
	defer enc.Close()

	// Threshold 0.01 accepts everything; real thresholds are swept offline
	// on the returned cosine scores. LSH off → exact top-K semantics.
	db := xordb.NewWithEncoder(enc,
		xordb.WithCapacity(1000),
		xordb.WithThreshold(0.01),
		xordb.WithLSH(false),
	)
	for _, qp := range Dataset {
		db.Set(qp.Cached, qp.Answer)
	}

	type result struct {
		score   float64
		correct bool // returned value matches the paired entry's answer
	}
	results := make([]result, len(Dataset))
	for i, qp := range Dataset {
		v, ok, sim := db.Get(qp.Lookup)
		if ok {
			s, _ := v.(string)
			results[i] = result{sim, s == qp.Answer}
		}
	}

	fmt.Println("\n══ STAGE E: full pipeline with cosine rerank ══")
	fmt.Printf("%-8s %6s %6s %6s %6s %6s %6s %6s %9s\n",
		"thresh", "TP", "FP", "FN", "TN", "prec%", "rec%", "f1%", "correct%")
	for _, th := range []float64{0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90} {
		var tp, fp, fn, tn, correct int
		for i, qp := range Dataset {
			hit := results[i].score >= th
			switch {
			case qp.ExpectHit && hit:
				tp++
				if results[i].correct {
					correct++
				}
			case !qp.ExpectHit && !hit:
				tn++
			case !qp.ExpectHit && hit:
				fp++
			default:
				fn++
			}
		}
		prec, rec, f1 := metrics(tp, fp, fn)
		correctPct := 0.0
		if tp > 0 {
			correctPct = 100 * float64(correct) / float64(tp)
		}
		fmt.Printf("%-8.2f %6d %6d %6d %6d %5.1f%% %5.1f%% %5.1f%% %8.1f%%\n",
			th, tp, fp, fn, tn, prec, rec, f1, correctPct)
	}

	// Category breakdown at the default threshold (0.75): correct outcomes
	// per category — hit for "match", rejection for "neg"/"hard-neg".
	fmt.Println("\n── category breakdown @ 0.75 ──")
	counts := map[string][2]int{} // category → [correct, total]
	for i, qp := range Dataset {
		hit := results[i].score >= 0.75
		ok := hit == qp.ExpectHit
		c := counts[qp.Category]
		if ok {
			c[0]++
		}
		c[1]++
		counts[qp.Category] = c
	}
	for _, cat := range []string{"match", "neg", "hard-neg"} {
		c := counts[cat]
		fmt.Printf("%-10s %d/%d\n", cat, c[0], c[1])
	}
}

// bestMatch returns the highest score over all cached entries and its index.
func bestMatch(n int, score func(j int) float64) (float64, int) {
	best, bestIdx := -1.0, -1
	for j := 0; j < n; j++ {
		if s := score(j); s > best {
			best, bestIdx = s, j
		}
	}
	return best, bestIdx
}

// sweepStage: for each lookup find the best-scoring cached entry, then sweep
// hit thresholds and print precision/recall/F1. Mirrors cache.Get semantics
// (best match above threshold = hit).
func sweepStage(bestFor func(i int) (float64, int), thresholds []float64) {
	n := len(Dataset)
	bestScores := make([]float64, n)
	for i := 0; i < n; i++ {
		bestScores[i], _ = bestFor(i)
	}

	fmt.Printf("%-8s %6s %6s %6s %6s %6s %6s %6s\n", "thresh", "TP", "FP", "FN", "TN", "prec%", "rec%", "f1%")
	for _, th := range thresholds {
		var tp, fp, fn, tn int
		for i, qp := range Dataset {
			hit := bestScores[i] >= th
			switch {
			case qp.ExpectHit && hit:
				tp++
			case !qp.ExpectHit && !hit:
				tn++
			case !qp.ExpectHit && hit:
				fp++
			default:
				fn++
			}
		}
		prec, rec, f1 := metrics(tp, fp, fn)
		fmt.Printf("%-8.2f %6d %6d %6d %6d %5.1f%% %5.1f%% %5.1f%%\n", th, tp, fp, fn, tn, prec, rec, f1)
	}
}

// printRecallAtK: fraction of expect_hit pairs whose own cached partner
// appears in the binary top-K. recall@K ≈ 99% ⇒ two-stage rerank recovers
// the float ceiling.
func printRecallAtK(n int, lookupBin, cachedBin []hdc.Vector) {
	ks := []int{1, 5, 10, 25, 50, 100}
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
			all[j] = scored{j, hdc.Similarity(lookupBin[i], cachedBin[j])}
		}
		sort.Slice(all, func(a, b int) bool { return all[a].sim > all[b].sim })
		rank := -1
		for r, s := range all {
			if s.idx == i {
				rank = r
				break
			}
		}
		for ki, k := range ks {
			if rank >= 0 && rank < k {
				hits[ki]++
			}
		}
	}

	for ki, k := range ks {
		fmt.Printf("recall@%-4d %5.1f%%  (%d/%d)\n", k, 100*float64(hits[ki])/float64(total), hits[ki], total)
	}
}

func printDistributions(n int, cachedEmb, lookupEmb [][]float32, cachedBin, lookupBin []hdc.Vector) {
	var fm, fn2, bm, bn []float64
	for i, qp := range Dataset {
		fs := cosine32(lookupEmb[i], cachedEmb[i])
		bs := hdc.Similarity(lookupBin[i], cachedBin[i])
		if qp.ExpectHit {
			fm = append(fm, fs)
			bm = append(bm, bs)
		} else {
			fn2 = append(fn2, fs)
			bn = append(bn, bs)
		}
	}
	fmt.Println("\n── paired score distributions (match vs non-match) ──")
	fmt.Printf("float  cosine:  match mean=%.4f  non-match mean=%.4f  gap=%.4f\n",
		meanf(fm), meanf(fn2), meanf(fm)-meanf(fn2))
	fmt.Printf("binary hamming: match mean=%.4f  non-match mean=%.4f  gap=%.4f\n",
		meanf(bm), meanf(bn), meanf(bm)-meanf(bn))
}

func cosine32(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}
