package embed

import (
	"strings"
	"unicode"
)

// Special token IDs for BERT uncased vocabulary.
const (
	clsTokenID = 101  // [CLS]
	sepTokenID = 102  // [SEP]
	unkTokenID = 100  // [UNK]
	padTokenID = 0    // [PAD]
)

// WordPieceTokenizer implements BERT-style WordPiece tokenization.
// It is safe for concurrent use after construction (read-only).
type WordPieceTokenizer struct {
	vocab    map[string]int32 // token string → token ID
	maxToken int              // longest token length (for subword search)
}

// NewWordPieceTokenizer creates a tokenizer from a vocabulary string where
// each line is a token and the line number (0-based) is its ID.
func NewWordPieceTokenizer(vocabText string) *WordPieceTokenizer {
	lines := strings.Split(vocabText, "\n")
	vocab := make(map[string]int32, len(lines))
	maxToken := 0
	for i, line := range lines {
		line = strings.TrimRight(line, "\r")
		if line == "" {
			continue
		}
		vocab[line] = int32(i)
		if len(line) > maxToken {
			maxToken = len(line)
		}
	}
	return &WordPieceTokenizer{vocab: vocab, maxToken: maxToken}
}

// TokenizeResult holds the output of tokenization.
type TokenizeResult struct {
	InputIDs      []int32 // token IDs including [CLS] and [SEP]
	AttentionMask []int32 // 1 for real tokens, 0 for padding
	TokenTypeIDs  []int32 // 0 for single-sentence input
}

// Tokenize converts text into BERT token IDs with [CLS] and [SEP] framing.
// If maxLen > 0, the output is truncated (before [SEP]) to fit within maxLen tokens.
func (t *WordPieceTokenizer) Tokenize(text string, maxLen int) TokenizeResult {
	// 1. Normalize: lowercase, strip accents, insert whitespace around punctuation.
	cleaned := t.preprocess(text)

	// 2. Split on whitespace to get initial words.
	words := strings.Fields(cleaned)

	// 3. WordPiece each word.
	ids := make([]int32, 0, len(words)*2+2)
	ids = append(ids, clsTokenID)

	for _, word := range words {
		ids = append(ids, t.wordPiece(word)...)
	}

	// 4. Truncate if needed (leave room for [SEP]).
	if maxLen > 0 && len(ids) >= maxLen {
		ids = ids[:maxLen-1]
	}
	ids = append(ids, sepTokenID)

	// 5. Build attention mask and token type IDs.
	n := len(ids)
	mask := make([]int32, n)
	typeIDs := make([]int32, n)
	for i := range mask {
		mask[i] = 1
	}

	return TokenizeResult{
		InputIDs:      ids,
		AttentionMask: mask,
		TokenTypeIDs:  typeIDs,
	}
}

// PadTo pads the TokenizeResult to exactly length n with [PAD] tokens.
// If already at or beyond n, no padding is added.
func (r *TokenizeResult) PadTo(n int) {
	for len(r.InputIDs) < n {
		r.InputIDs = append(r.InputIDs, padTokenID)
		r.AttentionMask = append(r.AttentionMask, 0)
		r.TokenTypeIDs = append(r.TokenTypeIDs, 0)
	}
}

// preprocess lowercases the text, strips accents, and inserts whitespace
// around punctuation characters so they become separate tokens.
func (t *WordPieceTokenizer) preprocess(text string) string {
	text = strings.ToLower(text)

	var b strings.Builder
	b.Grow(len(text) + 32)

	for _, r := range text {
		if unicode.In(r, unicode.Mn) {
			// Strip combining marks (accents).
			continue
		}
		if isPunctuation(r) {
			b.WriteByte(' ')
			b.WriteRune(r)
			b.WriteByte(' ')
		} else if unicode.IsSpace(r) || isControl(r) {
			b.WriteByte(' ')
		} else {
			b.WriteRune(r)
		}
	}

	return b.String()
}

// wordPiece splits a single whitespace-delimited word into WordPiece sub-tokens.
// Returns a slice of token IDs.
func (t *WordPieceTokenizer) wordPiece(word string) []int32 {
	if _, ok := t.vocab[word]; ok {
		return []int32{t.vocab[word]}
	}

	runes := []rune(word)
	ids := make([]int32, 0, 4)
	start := 0

	for start < len(runes) {
		end := len(runes)
		if end-start > t.maxToken {
			end = start + t.maxToken
		}

		found := false
		for end > start {
			substr := string(runes[start:end])
			if start > 0 {
				substr = "##" + substr
			}
			if id, ok := t.vocab[substr]; ok {
				ids = append(ids, id)
				start = end
				found = true
				break
			}
			end--
		}

		if !found {
			// Character not in vocab — emit [UNK] for the whole word.
			return []int32{unkTokenID}
		}
	}

	return ids
}

// isPunctuation checks if a rune is a punctuation character in the BERT sense.
func isPunctuation(r rune) bool {
	if (r >= 33 && r <= 47) || (r >= 58 && r <= 64) ||
		(r >= 91 && r <= 96) || (r >= 123 && r <= 126) {
		return true
	}
	return unicode.IsPunct(r)
}

// isControl checks if a rune is a control character (excluding whitespace).
func isControl(r rune) bool {
	if r == '\t' || r == '\n' || r == '\r' {
		return false
	}
	return unicode.IsControl(r)
}
