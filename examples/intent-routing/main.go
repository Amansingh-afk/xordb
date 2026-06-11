// Intent routing with xordb: register example phrases per intent, route
// queries to the nearest match. Uses the built-in n-gram encoder (character
// overlap, no synonyms); swap in xordb/embed MiniLM for semantic matching.
//
// Run: go run ./examples/intent-routing
package main

import (
	"fmt"

	"github.com/Amansingh-afk/xordb"
)

func main() {
	db := xordb.New(
		xordb.WithThreshold(0.55), // n-gram scores: matches ~0.6+, unrelated ~0.5
		xordb.WithCapacity(256),
	)

	// each example phrase maps to its intent label
	intents := map[string][]string{
		"greeting":     {"hello", "hi", "hey there", "good morning", "howdy"},
		"farewell":     {"goodbye", "bye", "see you later", "take care", "good night"},
		"help":         {"I need help", "can you help me", "support", "assist me", "I'm stuck"},
		"order_status": {"where is my order", "track my package", "order status", "shipping update"},
		"refund":       {"I want a refund", "give me my money back", "return this item", "cancel my order"},
	}
	for name, examples := range intents {
		for _, ex := range examples {
			db.Set(ex, name)
		}
	}

	queries := []string{
		"hey",
		"I want my money back",
		"where is my package",
		"can someone help me out",
		"see ya later",
		"what's the status of my delivery",
		"good evening",
		"I'd like to return my purchase",
		"quantum entanglement of neutrinos", // should miss
	}

	fmt.Printf("%-40s %-15s %s\n", "query", "intent", "score")
	fmt.Println("──────────────────────────────────────── ─────────────── ──────")
	for _, q := range queries {
		v, ok, sim := db.Get(q)
		intent := "(unknown)"
		if ok {
			intent = v.(string)
		}
		fmt.Printf("%-40s %-15s %.4f\n", q, intent, sim)
	}
}
