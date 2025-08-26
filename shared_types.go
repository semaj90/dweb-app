// shared_types.go
// Canonical shared request/response structs reused by multiple tagged binaries.
// Keeping them in package main avoids import cycles while code is flattened.
package main

// RAGRequest represents a retrieval augmented generation query.
type RAGRequest struct {
    Query      string                 `json:"query"`
    UserID     string                 `json:"user_id,omitempty"`
    CaseID     string                 `json:"case_id,omitempty"`
    Context    map[string]interface{} `json:"context,omitempty"`
    MaxResults int                    `json:"max_results,omitempty"`
}

// AIRequest represents a generic AI generation / inference prompt.
type AIRequest struct {
    Prompt      string                 `json:"prompt"`
    Model       string                 `json:"model,omitempty"`
    Context     map[string]interface{} `json:"context,omitempty"`
    MaxTokens   int                    `json:"max_tokens,omitempty"`
    Temperature float64                `json:"temperature,omitempty"`
}
