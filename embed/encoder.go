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
	// MiniLM-L6-v2 produces 384-dimensional float32 embeddings.
	miniLMEmbDims = 384

	// Default max sequence length for tokenization (BERT standard).
	defaultMaxSeqLen = 128

	// Default binary vector dimensionality for HDC projection.
	defaultBinaryDims = 10_000

	// Default projection seed for reproducibility.
	defaultProjectionSeed = 0xDB_CAFE
)

// MiniLMEncoder implements hdc.Encoder using a local MiniLM-L6-v2 ONNX model.
// It tokenizes input text with WordPiece, runs ONNX inference to get 384-dim
// float32 embeddings, then projects them to binary hdc.Vector via random
// hyperplane LSH.
//
// Thread-safe after construction.
type MiniLMEncoder struct {
	mu        sync.Mutex
	session   *ort.DynamicAdvancedSession
	tokenizer *WordPieceTokenizer
	projector *Projector
	maxSeqLen int
	binaryDims int
}

// EncoderOption configures a MiniLMEncoder.
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

// WithModelPath sets the path to the ONNX model file.
// If not set, the encoder looks in standard locations (see DefaultModelPath).
func WithModelPath(path string) EncoderOption {
	return func(c *encoderConfig) { c.modelPath = path }
}

// WithMaxSeqLen sets the maximum token sequence length. Default: 128.
// Longer inputs are truncated. Must be > 2 (to fit [CLS] and [SEP]).
func WithMaxSeqLen(n int) EncoderOption {
	return func(c *encoderConfig) { c.maxSeqLen = n }
}

// WithBinaryDims sets the output binary vector dimensionality. Default: 10000.
func WithBinaryDims(dims int) EncoderOption {
	return func(c *encoderConfig) { c.binaryDims = dims }
}

// WithProjectionSeed sets the seed for random hyperplane generation. Default: 0xXDB_CAFE.
// Using the same seed ensures deterministic projection across restarts.
func WithProjectionSeed(seed uint64) EncoderOption {
	return func(c *encoderConfig) { c.projectionSeed = seed }
}

// NewMiniLMEncoder creates a MiniLMEncoder.
//
// The ONNX runtime shared library must be available on the system. Call
// InitONNXRuntime before creating an encoder, or it will be initialized
// automatically with default settings.
//
// Returns an error if the model file is not found or ONNX session creation fails.
func NewMiniLMEncoder(opts ...EncoderOption) (*MiniLMEncoder, error) {
	cfg := defaultEncoderConfig()
	for _, opt := range opts {
		opt(&cfg)
	}

	if cfg.maxSeqLen < 3 {
		return nil, fmt.Errorf("embed: maxSeqLen must be >= 3, got %d", cfg.maxSeqLen)
	}

	// Resolve model path.
	modelPath := cfg.modelPath
	if modelPath == "" {
		var err error
		modelPath, err = DefaultModelPath()
		if err != nil {
			return nil, fmt.Errorf("embed: model not found: %w (use xordb-model download or WithModelPath)", err)
		}
	}
	if _, err := os.Stat(modelPath); err != nil {
		return nil, fmt.Errorf("embed: model file not accessible: %w", err)
	}

	// Initialize ONNX Runtime if not already done.
	if err := ensureONNXRuntime(); err != nil {
		return nil, fmt.Errorf("embed: ONNX runtime init failed: %w", err)
	}

	// Create ONNX session with dynamic axes.
	inputNames := []string{"input_ids", "attention_mask", "token_type_ids"}
	outputNames := []string{"last_hidden_state"}

	session, err := ort.NewDynamicAdvancedSession(
		modelPath,
		inputNames,
		outputNames,
		nil, // session options
	)
	if err != nil {
		return nil, fmt.Errorf("embed: failed to create ONNX session: %w", err)
	}

	tokenizer := NewWordPieceTokenizer(vocabData)
	projector := NewProjector(miniLMEmbDims, cfg.binaryDims, cfg.projectionSeed)

	return &MiniLMEncoder{
		session:    session,
		tokenizer:  tokenizer,
		projector:  projector,
		maxSeqLen:  cfg.maxSeqLen,
		binaryDims: cfg.binaryDims,
	}, nil
}

// Encode implements hdc.Encoder. It tokenizes the text, runs ONNX inference,
// applies mean pooling, L2-normalizes the embedding, and projects it to a
// binary hdc.Vector.
func (e *MiniLMEncoder) Encode(text string) hdc.Vector {
	emb, err := e.Embed(text)
	if err != nil {
		// hdc.Encoder interface doesn't return errors — return a zero vector.
		// This matches the behavior of the n-gram encoder for empty input.
		return hdc.New(e.binaryDims)
	}
	return e.projector.Project(emb)
}

// Embed returns the raw 384-dimensional float32 embedding for the given text.
// This is useful for debugging or for users who want to do their own projection.
func (e *MiniLMEncoder) Embed(text string) ([]float32, error) {
	// 1. Tokenize.
	tokens := e.tokenizer.Tokenize(text, e.maxSeqLen)
	seqLen := len(tokens.InputIDs)
	tokens.PadTo(e.maxSeqLen)

	// 2. Create ONNX tensors.
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

	// Output: [1, seq_len, 384]
	outputShape := ort.NewShape(1, int64(e.maxSeqLen), miniLMEmbDims)
	output, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return nil, fmt.Errorf("embed: creating output tensor: %w", err)
	}
	defer output.Destroy()

	// 3. Run inference.
	e.mu.Lock()
	err = e.session.Run(
		[]ort.ArbitraryTensor{inputIDs, attentionMask, tokenTypeIDs},
		[]ort.ArbitraryTensor{output},
	)
	e.mu.Unlock()
	if err != nil {
		return nil, fmt.Errorf("embed: ONNX inference failed: %w", err)
	}

	// 4. Mean pooling over non-padding tokens.
	outputData := output.GetData()
	embedding := meanPool(outputData, seqLen, e.maxSeqLen, miniLMEmbDims)

	// 5. L2 normalize.
	l2Normalize(embedding)

	return embedding, nil
}

// Close releases ONNX session resources. The encoder must not be used after Close.
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

// meanPool computes the mean of token embeddings, excluding padding tokens.
// data is [1, maxSeqLen, embDims], we average over tokens [0, seqLen).
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

// l2Normalize normalizes a vector to unit length in-place.
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

// castInt32ToInt64 converts a slice of int32 to int64 (ONNX Runtime expects int64).
func castInt32ToInt64(in []int32) []int64 {
	out := make([]int64, len(in))
	for i, v := range in {
		out[i] = int64(v)
	}
	return out
}

// ── ONNX Runtime initialization ──────────────────────────────────────────────

var ortOnce sync.Once
var ortErr error

// ensureONNXRuntime initializes the ONNX Runtime library if not already done.
func ensureONNXRuntime() error {
	ortOnce.Do(func() {
		// The library will look for the shared library in standard paths.
		// Users can set ORT_LIB_PATH or call ort.SetSharedLibraryPath before this.
		ortErr = ort.InitializeEnvironment()
	})
	return ortErr
}

// DestroyONNXRuntime cleans up the ONNX Runtime environment.
// Call this at application shutdown if you want clean resource release.
func DestroyONNXRuntime() error {
	return ort.DestroyEnvironment()
}

// ── Model path resolution ────────────────────────────────────────────────────

const modelFileName = "all-MiniLM-L6-v2.onnx"

// DefaultModelPath returns the default path where the ONNX model is expected.
// It checks the following locations in order:
//  1. $XORDB_MODEL_PATH (if set)
//  2. $XDG_DATA_HOME/xordb/models/all-MiniLM-L6-v2.onnx
//  3. ~/.local/share/xordb/models/all-MiniLM-L6-v2.onnx
//
// Returns the first path that exists, or an error if none is found.
func DefaultModelPath() (string, error) {
	// 1. Environment variable override.
	if p := os.Getenv("XORDB_MODEL_PATH"); p != "" {
		if _, err := os.Stat(p); err == nil {
			return p, nil
		}
	}

	// 2. XDG data directory.
	candidates := modelCandidatePaths()
	for _, p := range candidates {
		if _, err := os.Stat(p); err == nil {
			return p, nil
		}
	}

	return "", fmt.Errorf("model not found in any of: $XORDB_MODEL_PATH, %v", candidates)
}

// ModelDir returns the directory where models should be stored.
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
