//go:build !experimental && !legacy

package main

// Fallback SearchResult definition for builds where main.go (guarded by experimental||legacy) is excluded.
// This keeps ranking_cache.go typed without requiring interface{} conversions.
type SearchResult struct {
	DocumentID  string  `json:"document_id"`
	Content     string  `json:"content"`
	Score       float64 `json:"score"`
	Highlighted string  `json:"highlighted,omitempty"`
}
