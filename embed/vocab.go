package embed

import _ "embed"

// BERT uncased WordPiece vocab (30,522 tokens, ~227KB).
//
//go:embed testdata/vocab.txt
var vocabData string
