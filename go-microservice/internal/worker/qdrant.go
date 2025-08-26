package worker

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"
)

type QdrantClient struct {
	baseURL    string
	collection string
	httpc      *http.Client
}

func NewQdrantClient(baseURL, collection string) *QdrantClient {
	return &QdrantClient{
		baseURL:    strings.TrimRight(baseURL, "/"),
		collection: collection,
		httpc: &http.Client{
			Timeout: 15 * time.Second,
		},
	}
}

type upsertPointBody struct {
	Points []point `json:"points"`
}

type point struct {
	ID      string                 `json:"id"`
	Vector  []float32              `json:"vector"`
	Payload map[string]interface{} `json:"payload,omitempty"`
}

func (c *QdrantClient) UpsertPoint(ownerID string, vector []float32, payload map[string]interface{}) error {
	body := upsertPointBody{
		Points: []point{
			{
				ID:      ownerID,
				Vector:  vector,
				Payload: payload,
			},
		},
	}
	
	b, _ := json.Marshal(body)
	url := fmt.Sprintf("%s/collections/%s/points?wait=true", c.baseURL, c.collection)
	
	req, err := http.NewRequestWithContext(context.Background(), "PUT", url, bytes.NewReader(b))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	
	resp, err := c.httpc.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	
	if resp.StatusCode >= 300 {
		return fmt.Errorf("qdrant upsert failed status=%d", resp.StatusCode)
	}
	return nil
}

func (c *QdrantClient) Close() {
	// HTTP client doesn't need explicit closing
}