package embed

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sync"

	ort "github.com/yalue/onnxruntime_go"

	"xordb/hdc"
)

const (
	miniLMEmbDims         = 384    // MiniLM-L6-v2 output dims
	defaultMaxSeqLen      = 128
	defaultBinaryDims     = 10_000
	defaultProjectionSeed = 0xDB_CAFE
)

// MiniLMEncoder — local MiniLM-L6-v2 via ONNX → 384-dim float → binary HDC vector.
// Thread-safe after construction.
type MiniLMEncoder struct {
	mu        sync.Mutex
	session   *ort.DynamicAdvancedSession
	tokenizer *WordPieceTokenizer
	projector *Projector
	maxSeqLen  int
	binaryDims int
}

type EncoderOption func(*encoderConfig)

type encoderConfig struct {
	modelPath      string
	maxSeqLen      int
	binaryDims     int
	projectionSeed uint64
}

func defaultEncoderConfig() encoderConfig {
	return encoderConfig{
		maxSeqLen:      defaultMaxSeqLen,
		binaryDims:     defaultBinaryDims,
		projectionSeed: defaultProjectionSeed,
	}
}

func WithModelPath(path string) EncoderOption {
	return func(c *encoderConfig) { c.modelPath = path }
}

func WithMaxSeqLen(n int) EncoderOption {
	return func(c *encoderConfig) { c.maxSeqLen = n }
}

func WithBinaryDims(dims int) EncoderOption {
	return func(c *encoderConfig) { c.binaryDims = dims }
}

func WithProjectionSeed(seed uint64) EncoderOption {
	return func(c *encoderConfig) { c.projectionSeed = seed }
}

// NewMiniLMEncoder creates the encoder. ONNX runtime must be available.
// Model path is auto-resolved if not set (see DefaultModelPath).
func NewMiniLMEncoder(opts ...EncoderOption) (*MiniLMEncoder, error) {
	cfg := defaultEncoderConfig()
	for _, opt := range opts {
		opt(&cfg)
	}

	if cfg.maxSeqLen < 3 {
		return nil, fmt.Errorf("embed: maxSeqLen must be >= 3, got %d", cfg.maxSeqLen)
	}

	modelPath := cfg.modelPath
	if modelPath == "" {
		var err error
		modelPath, err = DefaultModelPath()
		if err != nil {
			return nil, fmt.Errorf("embed: model not found: %w (run: xordb-model download)", err)
		}
	}
	if _, err := os.Stat(modelPath); err != nil {
		return nil, fmt.Errorf("embed: model file not accessible: %w", err)
	}

	if err := ensureONNXRuntime(); err != nil {
		return nil, fmt.Errorf("embed: ONNX runtime init failed: %w", err)
	}

	session, err := ort.NewDynamicAdvancedSession(
		modelPath,
		[]string{"input_ids", "attention_mask", "token_type_ids"},
		[]string{"last_hidden_state"},
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("embed: failed to create ONNX session: %w", err)
	}

	return &MiniLMEncoder{
		session:    session,
		tokenizer:  NewWordPieceTokenizer(vocabData),
		projector:  NewProjector(miniLMEmbDims, cfg.binaryDims, cfg.projectionSeed),
		maxSeqLen:  cfg.maxSeqLen,
		binaryDims: cfg.binaryDims,
	}, nil
}

// Encode implements hdc.Encoder. Error → zero vector (interface mein error nahi hai).
func (e *MiniLMEncoder) Encode(text string) hdc.Vector {
	emb, err := e.Embed(text)
	if err != nil {
		return hdc.New(e.binaryDims)
	}
	return e.projector.Project(emb)
}

// Embed returns the raw 384-dim float32 embedding (useful for debugging).
func (e *MiniLMEncoder) Embed(text string) ([]float32, error) {
	tokens := e.tokenizer.Tokenize(text, e.maxSeqLen)
	seqLen := len(tokens.InputIDs)
	tokens.PadTo(e.maxSeqLen)

	shape := ort.NewShape(1, int64(e.maxSeqLen))

	inputIDs, err := ort.NewTensor(shape, castInt32ToInt64(tokens.InputIDs))
	if err != nil {
		return nil, fmt.Errorf("embed: creating input_ids tensor: %w", err)
	}
	defer inputIDs.Destroy()

	attentionMask, err := ort.NewTensor(shape, castInt32ToInt64(tokens.AttentionMask))
	if err != nil {
		return nil, fmt.Errorf("embed: creating attention_mask tensor: %w", err)
	}
	defer attentionMask.Destroy()

	tokenTypeIDs, err := ort.NewTensor(shape, castInt32ToInt64(tokens.TokenTypeIDs))
	if err != nil {
		return nil, fmt.Errorf("embed: creating token_type_ids tensor: %w", err)
	}
	defer tokenTypeIDs.Destroy()

	outputShape := ort.NewShape(1, int64(e.maxSeqLen), miniLMEmbDims)
	output, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return nil, fmt.Errorf("embed: creating output tensor: %w", err)
	}
	defer output.Destroy()

	e.mu.Lock()
	err = e.session.Run(
		[]ort.ArbitraryTensor{inputIDs, attentionMask, tokenTypeIDs},
		[]ort.ArbitraryTensor{output},
	)
	e.mu.Unlock()
	if err != nil {
		return nil, fmt.Errorf("embed: ONNX inference failed: %w", err)
	}

	outputData := output.GetData()
	embedding := meanPool(outputData, seqLen, e.maxSeqLen, miniLMEmbDims)
	l2Normalize(embedding)

	return embedding, nil
}

func (e *MiniLMEncoder) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.session != nil {
		err := e.session.Destroy()
		e.session = nil
		return err
	}
	return nil
}

// meanPool — average over non-padding tokens.
func meanPool(data []float32, seqLen, maxSeqLen, embDims int) []float32 {
	result := make([]float32, embDims)
	if seqLen == 0 {
		return result
	}

	for t := 0; t < seqLen; t++ {
		offset := t * embDims
		for d := 0; d < embDims; d++ {
			result[d] += data[offset+d]
		}
	}

	scale := 1.0 / float32(seqLen)
	for d := range result {
		result[d] *= scale
	}
	return result
}

func l2Normalize(v []float32) {
	var norm float64
	for _, x := range v {
		norm += float64(x) * float64(x)
	}
	norm = math.Sqrt(norm)
	if norm > 0 {
		scale := float32(1.0 / norm)
		for i := range v {
			v[i] *= scale
		}
	}
}

func castInt32ToInt64(in []int32) []int64 {
	out := make([]int64, len(in))
	for i, v := range in {
		out[i] = int64(v)
	}
	return out
}

// ── ONNX Runtime init ────────────────────────────────────────────────────────

var ortOnce sync.Once
var ortErr error

func ensureONNXRuntime() error {
	ortOnce.Do(func() {
		ortErr = ort.InitializeEnvironment()
	})
	return ortErr
}

func DestroyONNXRuntime() error {
	return ort.DestroyEnvironment()
}

// ── Model path resolution ────────────────────────────────────────────────────

const modelFileName = "all-MiniLM-L6-v2.onnx"

// DefaultModelPath checks: $XORDB_MODEL_PATH → $XDG_DATA_HOME/xordb/models/ → ~/.local/share/xordb/models/
func DefaultModelPath() (string, error) {
	if p := os.Getenv("XORDB_MODEL_PATH"); p != "" {
		if _, err := os.Stat(p); err == nil {
			return p, nil
		}
	}

	candidates := modelCandidatePaths()
	for _, p := range candidates {
		if _, err := os.Stat(p); err == nil {
			return p, nil
		}
	}

	return "", fmt.Errorf("model not found in any of: $XORDB_MODEL_PATH, %v", candidates)
}

func ModelDir() string {
	dataDir := os.Getenv("XDG_DATA_HOME")
	if dataDir == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			home = "."
		}
		if runtime.GOOS == "darwin" {
			dataDir = filepath.Join(home, "Library", "Application Support")
		} else {
			dataDir = filepath.Join(home, ".local", "share")
		}
	}
	return filepath.Join(dataDir, "xordb", "models")
}

func modelCandidatePaths() []string {
	return []string{
		filepath.Join(ModelDir(), modelFileName),
	}
}
