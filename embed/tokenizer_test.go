package embed

import (
	"testing"
)

func newTestTokenizer() *WordPieceTokenizer {
	return NewWordPieceTokenizer(vocabData)
}

// ── construction ──────────────────────────────────────────────────────────────

func TestNewWordPieceTokenizer_LoadsVocab(t *testing.T) {
	tok := newTestTokenizer()
	if len(tok.vocab) < 30000 {
		t.Fatalf("expected ~30522 vocab entries, got %d", len(tok.vocab))
	}
}

func TestNewWordPieceTokenizer_SpecialTokens(t *testing.T) {
	tok := newTestTokenizer()
	checks := map[string]int32{
		"[PAD]": 0,
		"[UNK]": 100,
		"[CLS]": 101,
		"[SEP]": 102,
	}
	for token, wantID := range checks {
		if id, ok := tok.vocab[token]; !ok {
			t.Fatalf("vocab missing %s", token)
		} else if id != wantID {
			t.Fatalf("want %s=%d, got %d", token, wantID, id)
		}
	}
}

// ── basic tokenization ───────────────────────────────────────────────────────

func TestTokenize_SimpleText(t *testing.T) {
	tok := newTestTokenizer()
	res := tok.Tokenize("hello world", 0)

	// Must start with [CLS] and end with [SEP]
	if res.InputIDs[0] != clsTokenID {
		t.Fatalf("first token must be [CLS]=%d, got %d", clsTokenID, res.InputIDs[0])
	}
	last := res.InputIDs[len(res.InputIDs)-1]
	if last != sepTokenID {
		t.Fatalf("last token must be [SEP]=%d, got %d", sepTokenID, last)
	}

	// "hello" and "world" are in BERT vocab
	if len(res.InputIDs) < 4 {
		t.Fatalf("expected at least 4 tokens ([CLS] hello world [SEP]), got %d", len(res.InputIDs))
	}
}

func TestTokenize_AttentionMask(t *testing.T) {
	tok := newTestTokenizer()
	res := tok.Tokenize("hello", 0)

	for i, m := range res.AttentionMask {
		if m != 1 {
			t.Fatalf("attention mask[%d] should be 1, got %d", i, m)
		}
	}
}

func TestTokenize_TokenTypeIDs_AllZero(t *testing.T) {
	tok := newTestTokenizer()
	res := tok.Tokenize("hello world", 0)

	for i, tt := range res.TokenTypeIDs {
		if tt != 0 {
			t.Fatalf("token type[%d] should be 0 for single sentence, got %d", i, tt)
		}
	}
}

// ── case normalization ────────────────────────────────────────────────────────

func TestTokenize_CaseInsensitive(t *testing.T) {
	tok := newTestTokenizer()
	r1 := tok.Tokenize("Hello World", 0)
	r2 := tok.Tokenize("hello world", 0)

	if len(r1.InputIDs) != len(r2.InputIDs) {
		t.Fatalf("case variants should produce same token count: %d vs %d",
			len(r1.InputIDs), len(r2.InputIDs))
	}
	for i := range r1.InputIDs {
		if r1.InputIDs[i] != r2.InputIDs[i] {
			t.Fatalf("token[%d] differs: %d vs %d", i, r1.InputIDs[i], r2.InputIDs[i])
		}
	}
}

// ── punctuation handling ──────────────────────────────────────────────────────

func TestTokenize_PunctuationSeparated(t *testing.T) {
	tok := newTestTokenizer()
	res := tok.Tokenize("hello, world!", 0)

	// Should have more tokens than "hello world" because , and ! are separate
	plain := tok.Tokenize("hello world", 0)
	if len(res.InputIDs) <= len(plain.InputIDs) {
		t.Fatal("punctuation should produce additional tokens")
	}
}

// ── WordPiece subword splitting ──────────────────────────────────────────────

func TestTokenize_SubwordSplitting(t *testing.T) {
	tok := newTestTokenizer()
	// "embeddings" should be split into subwords like "em", "##bed", "##ding", "##s"
	res := tok.Tokenize("embeddings", 0)

	// At minimum: [CLS] + at least 1 token + [SEP]
	if len(res.InputIDs) < 3 {
		t.Fatalf("expected at least 3 tokens, got %d", len(res.InputIDs))
	}

	// Should not contain [UNK] for a common English word
	for _, id := range res.InputIDs {
		if id == unkTokenID {
			t.Fatal("common English word should not produce [UNK]")
		}
	}
}

// ── truncation ────────────────────────────────────────────────────────────────

func TestTokenize_Truncation(t *testing.T) {
	tok := newTestTokenizer()
	res := tok.Tokenize("the quick brown fox jumps over the lazy dog", 8)

	if len(res.InputIDs) > 8 {
		t.Fatalf("truncation to maxLen=8 failed, got %d tokens", len(res.InputIDs))
	}

	// Must still end with [SEP]
	last := res.InputIDs[len(res.InputIDs)-1]
	if last != sepTokenID {
		t.Fatalf("truncated sequence must end with [SEP], got %d", last)
	}
}

func TestTokenize_NoTruncation_WhenZero(t *testing.T) {
	tok := newTestTokenizer()
	res := tok.Tokenize("the quick brown fox jumps over the lazy dog", 0)

	// Should have all tokens
	if len(res.InputIDs) < 10 {
		t.Fatalf("without truncation should have many tokens, got %d", len(res.InputIDs))
	}
}

// ── padding ───────────────────────────────────────────────────────────────────

func TestPadTo(t *testing.T) {
	tok := newTestTokenizer()
	res := tok.Tokenize("hello", 0)
	origLen := len(res.InputIDs)

	res.PadTo(32)

	if len(res.InputIDs) != 32 {
		t.Fatalf("want padded length=32, got %d", len(res.InputIDs))
	}
	// Padding tokens should be [PAD]=0
	for i := origLen; i < 32; i++ {
		if res.InputIDs[i] != padTokenID {
			t.Fatalf("padding token[%d] should be %d, got %d", i, padTokenID, res.InputIDs[i])
		}
		if res.AttentionMask[i] != 0 {
			t.Fatalf("padding mask[%d] should be 0, got %d", i, res.AttentionMask[i])
		}
	}
}

func TestPadTo_AlreadyLongEnough(t *testing.T) {
	tok := newTestTokenizer()
	res := tok.Tokenize("hello world", 0)
	origLen := len(res.InputIDs)

	res.PadTo(2) // shorter than actual — should not truncate
	if len(res.InputIDs) != origLen {
		t.Fatalf("PadTo shorter than current should not change length: %d vs %d", origLen, len(res.InputIDs))
	}
}

// ── empty input ───────────────────────────────────────────────────────────────

func TestTokenize_EmptyString(t *testing.T) {
	tok := newTestTokenizer()
	res := tok.Tokenize("", 0)

	// Should still have [CLS] [SEP]
	if len(res.InputIDs) != 2 {
		t.Fatalf("empty input should produce [CLS][SEP], got %d tokens", len(res.InputIDs))
	}
	if res.InputIDs[0] != clsTokenID || res.InputIDs[1] != sepTokenID {
		t.Fatal("empty input must produce [CLS][SEP]")
	}
}

// ── benchmarks ────────────────────────────────────────────────────────────────

func BenchmarkTokenize_Short(b *testing.B) {
	tok := newTestTokenizer()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tok.Tokenize("what is the capital of india", 128)
	}
}

func BenchmarkTokenize_Medium(b *testing.B) {
	tok := newTestTokenizer()
	text := "the quick brown fox jumps over the lazy dog and the cow jumped over the moon"
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tok.Tokenize(text, 128)
	}
}
