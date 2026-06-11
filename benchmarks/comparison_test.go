package benchmarks

import (
	"context"
	"fmt"
	"os"
	"runtime"
	"testing"
	"time"

	hdc "github.com/Amansingh-afk/hdc-go"
	"github.com/Amansingh-afk/xordb"
	"github.com/Amansingh-afk/xordb/embed"
	chromem "github.com/philippgille/chromem-go"
	ort "github.com/yalue/onnxruntime_go"
)

// Head-to-head vs chromem-go, both fed the same MiniLM embeddings.
// Store-side timings use precomputed embeddings so the (identical) ONNX
// cost doesn't mask store differences.

// memoEncoder serves precomputed embeddings; projection matches the MiniLM
// encoder defaults.
type memoEncoder struct {
	embs map[string][]float32
	proj *hdc.Projector
}

func newMemoEncoder(embs map[string][]float32) *memoEncoder {
	return &memoEncoder{embs: embs, proj: hdc.NewProjector(384, 1024, 0xDB_CAFE)}
}

func (m *memoEncoder) Embed(text string) ([]float32, error) {
	e, ok := m.embs[text]
	if !ok {
		return nil, fmt.Errorf("memoEncoder: no embedding for %q", text)
	}
	return e, nil
}
func (m *memoEncoder) Project(emb []float32) hdc.Vector { return m.proj.ProjectFloat(emb) }
func (m *memoEncoder) Encode(text string) hdc.Vector {
	e, err := m.Embed(text)
	if err != nil {
		return hdc.New(1024)
	}
	return m.Project(e)
}

func TestComparison_Accuracy(t *testing.T) {
	if p := os.Getenv("ORT_LIB_PATH"); p != "" {
		ort.SetSharedLibraryPath(p)
	}
	enc, err := embed.NewMiniLMEncoder()
	if err != nil {
		t.Skipf("MiniLM encoder not available: %v", err)
	}
	defer enc.Close()

	embs := make(map[string][]float32, 2*len(Dataset))
	for _, qp := range Dataset {
		for _, s := range []string{qp.Cached, qp.Lookup} {
			if _, ok := embs[s]; !ok {
				e, err := enc.Embed(s)
				if err != nil {
					t.Fatal(err)
				}
				embs[s] = e
			}
		}
	}

	menc := newMemoEncoder(embs)
	db := xordb.NewWithEncoder(menc,
		xordb.WithCapacity(1000),
		xordb.WithThreshold(0.01),
		xordb.WithLSH(false),
	)
	for _, qp := range Dataset {
		db.Set(qp.Cached, qp.Answer)
	}
	xordbScores := make([]float64, len(Dataset))
	for i, qp := range Dataset {
		if _, ok, sim := db.Get(qp.Lookup); ok {
			xordbScores[i] = sim
		}
	}

	ctx := context.Background()
	cdb := chromem.NewDB()
	coll, err := cdb.CreateCollection("bench", nil, func(_ context.Context, text string) ([]float32, error) {
		return menc.Embed(text)
	})
	if err != nil {
		t.Fatal(err)
	}
	docs := make([]chromem.Document, len(Dataset))
	for i, qp := range Dataset {
		docs[i] = chromem.Document{ID: fmt.Sprint(i), Content: qp.Cached, Embedding: embs[qp.Cached]}
	}
	if err := coll.AddDocuments(ctx, docs, runtime.NumCPU()); err != nil {
		t.Fatal(err)
	}
	chromemScores := make([]float64, len(Dataset))
	for i, qp := range Dataset {
		res, err := coll.QueryEmbedding(ctx, embs[qp.Lookup], 1, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		if len(res) > 0 {
			chromemScores[i] = float64(res[0].Similarity)
		}
	}

	fmt.Println("\n══ ACCURACY: xordb (rerank) vs chromem-go, same MiniLM embeddings ══")
	for name, scores := range map[string][]float64{"xordb": xordbScores, "chromem-go": chromemScores} {
		fmt.Printf("\n── %s ──\n", name)
		fmt.Printf("%-8s %6s %6s %6s %6s %6s %6s %6s\n", "thresh", "TP", "FP", "FN", "TN", "prec%", "rec%", "f1%")
		for _, th := range []float64{0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90} {
			var tp, fp, fn, tn int
			for i, qp := range Dataset {
				hit := scores[i] >= th
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
}

func TestComparison_Perf(t *testing.T) {
	if p := os.Getenv("ORT_LIB_PATH"); p != "" {
		ort.SetSharedLibraryPath(p)
	}
	enc, err := embed.NewMiniLMEncoder()
	if err != nil {
		t.Skipf("MiniLM encoder not available: %v", err)
	}
	defer enc.Close()

	const nDocs = 5000
	const nQueries = 200

	topics := []string{"billing", "shipping", "returns", "passwords", "payments",
		"invoices", "subscriptions", "devices", "warranty", "accounts"}
	docTexts := make([]string, nDocs)
	for i := range docTexts {
		docTexts[i] = fmt.Sprintf("customer question %d about %s and how to resolve issue %d quickly",
			i, topics[i%len(topics)], i%97)
	}
	queryTexts := make([]string, nQueries)
	for i := range queryTexts {
		queryTexts[i] = fmt.Sprintf("how do I resolve %s issue %d as a customer", topics[i%len(topics)], i%97)
	}

	embStart := time.Now()
	embs := make(map[string][]float32, nDocs+nQueries)
	for _, s := range append(append([]string{}, docTexts...), queryTexts...) {
		e, err := enc.Embed(s)
		if err != nil {
			t.Fatal(err)
		}
		embs[s] = e
	}
	embedAvg := time.Since(embStart) / time.Duration(nDocs+nQueries)

	fmt.Printf("\n══ PERF: %d docs, %d queries (store-side; shared MiniLM embed cost = %v/op) ══\n",
		nDocs, nQueries, embedAvg.Round(time.Microsecond))

	heap := func() uint64 {
		runtime.GC()
		var ms runtime.MemStats
		runtime.ReadMemStats(&ms)
		return ms.HeapAlloc
	}

	menc := newMemoEncoder(embs)
	h0 := heap()
	db := xordb.NewWithEncoder(menc,
		xordb.WithCapacity(nDocs),
		xordb.WithThreshold(0.3),
	)
	start := time.Now()
	for i, s := range docTexts {
		db.Set(s, i)
	}
	xIngest := time.Since(start)
	xMem := heap() - h0

	start = time.Now()
	xHits := 0
	for _, q := range queryTexts {
		if _, ok, _ := db.Get(q); ok {
			xHits++
		}
	}
	xQuery := time.Since(start) / nQueries

	ctx := context.Background()
	h0 = heap()
	cdb := chromem.NewDB()
	coll, err := cdb.CreateCollection("bench", nil, func(_ context.Context, text string) ([]float32, error) {
		return menc.Embed(text)
	})
	if err != nil {
		t.Fatal(err)
	}
	start = time.Now()
	docs := make([]chromem.Document, nDocs)
	for i, s := range docTexts {
		// deep-copy so the stored vector counts in chromem's heap delta
		docs[i] = chromem.Document{ID: fmt.Sprint(i), Content: s, Embedding: append([]float32(nil), embs[s]...)}
	}
	if err := coll.AddDocuments(ctx, docs, 1); err != nil { // serial, same as xordb's Set
		t.Fatal(err)
	}
	cIngest := time.Since(start)
	cMem := heap() - h0

	start = time.Now()
	cHits := 0
	for _, q := range queryTexts {
		res, err := coll.QueryEmbedding(ctx, embs[q], 1, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		if len(res) > 0 && res[0].Similarity >= 0.3 {
			cHits++
		}
	}
	cQuery := time.Since(start) / nQueries

	fmt.Printf("%-12s %14s %16s %14s %8s\n", "store", "ingest total", "query (store)", "mem/entry", "hits")
	fmt.Printf("%-12s %14v %16v %12dB %8d\n", "xordb", xIngest.Round(time.Millisecond), xQuery.Round(time.Microsecond), xMem/nDocs, xHits)
	fmt.Printf("%-12s %14v %16v %12dB %8d\n", "chromem-go", cIngest.Round(time.Millisecond), cQuery.Round(time.Microsecond), cMem/nDocs, cHits)
	fmt.Printf("\nEnd-to-end query latency = store query + embed (%v): xordb %v, chromem-go %v\n",
		embedAvg.Round(time.Microsecond),
		(xQuery + embedAvg).Round(time.Microsecond),
		(cQuery + embedAvg).Round(time.Microsecond))
}
