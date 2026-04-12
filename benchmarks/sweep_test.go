package benchmarks

import (
	"fmt"
	"testing"

	hdc "github.com/Amansingh-afk/hdc-go"
	"github.com/Amansingh-afk/xordb"
)

// TestSweep_NGram_ThresholdAndDims runs a parameter sweep over threshold and
// dimensionality for the n-gram encoder. Prints a compact table of results.
func TestSweep_NGram_ThresholdAndDims(t *testing.T) {
	thresholds := []float64{0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.82}
	dims := []int{5000, 10000, 20000, 50000}

	fmt.Println()
	fmt.Printf("%-8s %-8s %6s %6s %6s %6s %6s %6s %6s\n",
		"dims", "thresh", "TP", "FP", "FN", "TN", "prec%", "rec%", "f1%")
	fmt.Println("──────── ──────── ────── ────── ────── ────── ────── ────── ──────")

	for _, d := range dims {
		for _, th := range thresholds {
			tp, fp, fn, tn := runNGramSweep(d, 3, th)
			prec, rec, f1 := metrics(tp, fp, fn)
			fmt.Printf("%-8d %-8.2f %6d %6d %6d %6d %5.1f%% %5.1f%% %5.1f%%\n",
				d, th, tp, fp, fn, tn, prec, rec, f1)
		}
		fmt.Println()
	}
}

// TestSweep_NGram_NGramSize tests different n-gram window sizes.
func TestSweep_NGram_NGramSize(t *testing.T) {
	ngramSizes := []int{2, 3, 4, 5}
	thresholds := []float64{0.55, 0.60, 0.65, 0.70, 0.75}

	fmt.Println()
	fmt.Printf("%-8s %-8s %6s %6s %6s %6s %6s %6s %6s\n",
		"ngram", "thresh", "TP", "FP", "FN", "TN", "prec%", "rec%", "f1%")
	fmt.Println("──────── ──────── ────── ────── ────── ────── ────── ────── ──────")

	for _, ng := range ngramSizes {
		for _, th := range thresholds {
			tp, fp, fn, tn := runNGramSweep(10000, ng, th)
			prec, rec, f1 := metrics(tp, fp, fn)
			fmt.Printf("%-8d %-8.2f %6d %6d %6d %6d %5.1f%% %5.1f%% %5.1f%%\n",
				ng, th, tp, fp, fn, tn, prec, rec, f1)
		}
		fmt.Println()
	}
}

// TestSweep_NGram_SimDistribution prints the similarity distribution for matches
// vs non-matches to help pick optimal thresholds.
func TestSweep_NGram_SimDistribution(t *testing.T) {
	enc := hdc.NewNGramEncoder(hdc.DefaultConfig())

	var matchSims, nonMatchSims []float64
	for _, qp := range Dataset {
		cachedVec := enc.Encode(qp.Cached)
		lookupVec := enc.Encode(qp.Lookup)
		sim := hdc.Similarity(cachedVec, lookupVec)

		if qp.ExpectHit {
			matchSims = append(matchSims, sim)
		} else {
			nonMatchSims = append(nonMatchSims, sim)
		}
	}

	// Print histogram buckets
	buckets := []struct{ lo, hi float64 }{
		{0.40, 0.45}, {0.45, 0.50}, {0.50, 0.55}, {0.55, 0.60},
		{0.60, 0.65}, {0.65, 0.70}, {0.70, 0.75}, {0.75, 0.80},
		{0.80, 0.85}, {0.85, 0.90}, {0.90, 0.95}, {0.95, 1.01},
	}

	fmt.Println()
	fmt.Printf("%-12s %8s %8s\n", "sim range", "matches", "non-match")
	fmt.Println("──────────── ──────── ────────")
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

	// Print stats
	fmt.Printf("\nMatch sims:     min=%.4f  max=%.4f  mean=%.4f  (n=%d)\n",
		minf(matchSims), maxf(matchSims), meanf(matchSims), len(matchSims))
	fmt.Printf("Non-match sims: min=%.4f  max=%.4f  mean=%.4f  (n=%d)\n",
		minf(nonMatchSims), maxf(nonMatchSims), meanf(nonMatchSims), len(nonMatchSims))
}

func runNGramSweep(dims, ngramSize int, threshold float64) (tp, fp, fn, tn int) {
	cfg := hdc.Config{
		Dims:           dims,
		NGramSize:      ngramSize,
		LongTextThresh: 200,
		ChunkSize:      128,
	}
	enc := hdc.NewNGramEncoder(cfg)

	db := xordb.NewWithEncoder(enc,
		xordb.WithThreshold(threshold),
		xordb.WithCapacity(1000),
	)

	for _, qp := range Dataset {
		db.Set(qp.Cached, qp.Answer)
	}

	for _, qp := range Dataset {
		_, ok, _ := db.Get(qp.Lookup)
		switch {
		case qp.ExpectHit && ok:
			tp++
		case !qp.ExpectHit && !ok:
			tn++
		case !qp.ExpectHit && ok:
			fp++
		case qp.ExpectHit && !ok:
			fn++
		}
	}
	return
}

func metrics(tp, fp, fn int) (prec, rec, f1 float64) {
	if tp+fp > 0 {
		prec = float64(tp) / float64(tp+fp) * 100
	}
	if tp+fn > 0 {
		rec = float64(tp) / float64(tp+fn) * 100
	}
	if prec+rec > 0 {
		f1 = 2 * prec * rec / (prec + rec)
	}
	return
}

func minf(s []float64) float64 {
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

func maxf(s []float64) float64 {
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

func meanf(s []float64) float64 {
	if len(s) == 0 {
		return 0
	}
	var sum float64
	for _, v := range s {
		sum += v
	}
	return sum / float64(len(s))
}
