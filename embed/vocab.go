package embed

import _ "embed"

// vocabData is the BERT uncased WordPiece vocabulary (30,522 tokens, ~227KB).
// Each line is a token; the line number (0-based) is the token ID.
//
//go:embed testdata/vocab.txt
var vocabData string
