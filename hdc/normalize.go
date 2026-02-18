package hdc

import (
	"strings"
	"unicode"
)

// normalizeSegment collapses whitespace, optionally strips punctuation.
func normalizeSegment(text string, stripPunct bool) string {
	var b strings.Builder
	b.Grow(len(text))
	prevSpace := false

	for _, r := range text {
		switch {
		case unicode.IsSpace(r):
			if !prevSpace {
				b.WriteByte(' ')
				prevSpace = true
			}
		case stripPunct && unicode.IsPunct(r):
			// hata do punctuation
		default:
			b.WriteRune(r)
			prevSpace = false
		}
	}

	return strings.TrimSpace(b.String())
}

// splitSentences splits on . ? ! and \n.
func splitSentences(text string) []string {
	var out []string
	var cur strings.Builder

	for _, r := range text {
		if r == '.' || r == '?' || r == '!' || r == '\n' {
			if s := strings.TrimSpace(cur.String()); s != "" {
				out = append(out, s)
			}
			cur.Reset()
		} else {
			cur.WriteRune(r)
		}
	}
	if s := strings.TrimSpace(cur.String()); s != "" {
		out = append(out, s)
	}
	return out
}
