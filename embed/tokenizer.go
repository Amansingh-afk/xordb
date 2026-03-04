package embed

import (
	"strings"
	"unicode"
)

// BERT uncased special tokens
const (
	clsTokenID = 101
	sepTokenID = 102
	unkTokenID = 100
	padTokenID = 0
)

// WordPieceTokenizer — BERT-style subword tokenization. Read-only after init.
type WordPieceTokenizer struct {
	vocab    map[string]int32
	maxToken int
}

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

type TokenizeResult struct {
	InputIDs      []int32
	AttentionMask []int32
	TokenTypeIDs  []int32
}

// Tokenize converts text into BERT token IDs with [CLS] and [SEP].
func (t *WordPieceTokenizer) Tokenize(text string, maxLen int) TokenizeResult {
	cleaned := t.preprocess(text)
	words := strings.Fields(cleaned)

	ids := make([]int32, 0, len(words)*2+2)
	ids = append(ids, clsTokenID)

	for _, word := range words {
		ids = append(ids, t.wordPiece(word)...)
	}

	if maxLen > 0 && len(ids) >= maxLen {
		ids = ids[:maxLen-1]
	}
	ids = append(ids, sepTokenID)

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

// PadTo pads to exactly n tokens.
func (r *TokenizeResult) PadTo(n int) {
	for len(r.InputIDs) < n {
		r.InputIDs = append(r.InputIDs, padTokenID)
		r.AttentionMask = append(r.AttentionMask, 0)
		r.TokenTypeIDs = append(r.TokenTypeIDs, 0)
	}
}

func (t *WordPieceTokenizer) preprocess(text string) string {
	text = strings.ToLower(text)

	var b strings.Builder
	b.Grow(len(text) + 32)

	for _, r := range text {
		if unicode.In(r, unicode.Mn) {
			continue // strip accents
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
			return []int32{unkTokenID}
		}
	}

	return ids
}

func isPunctuation(r rune) bool {
	if (r >= 33 && r <= 47) || (r >= 58 && r <= 64) ||
		(r >= 91 && r <= 96) || (r >= 123 && r <= 126) {
		return true
	}
	return unicode.IsPunct(r)
}

func isControl(r rune) bool {
	if r == '\t' || r == '\n' || r == '\r' {
		return false
	}
	return unicode.IsControl(r)
}
