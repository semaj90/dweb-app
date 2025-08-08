package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

type OllamaClient struct {
	baseURL string
	client  *http.Client
}

func NewOllamaClient() *OllamaClient {
	return &OllamaClient{
		baseURL: "http://localhost:11434",
		client:  &http.Client{Timeout: 60 * time.Second},
	}
}

func (o *OllamaClient) Embed(text string) ([]float32, error) {
	req := map[string]interface{}{
		"model":  "nomic-embed-text",
		"prompt": text,
	}
	
	body, _ := json.Marshal(req)
	resp, err := o.client.Post(o.baseURL+"/api/embeddings", "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	
	var result struct {
		Embedding []float32 `json:"embedding"`
	}
	json.NewDecoder(resp.Body).Decode(&result)
	return result.Embedding, nil
}

func (o *OllamaClient) Generate(prompt string) (string, error) {
	req := map[string]interface{}{
		"model":  "llama3",
		"prompt": prompt,
		"stream": false,
	}
	
	body, _ := json.Marshal(req)
	resp, err := o.client.Post(o.baseURL+"/api/generate", "application/json", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	
	var result struct {
		Response string `json:"response"`
	}
	json.NewDecoder(resp.Body).Decode(&result)
	return result.Response, nil
}

// Integration endpoints
func setupOllamaHandlers(r *gin.Engine) {
	ollama := NewOllamaClient()
	
	r.POST("/embed", func(c *gin.Context) {
		var req struct {
			Text string `json:"text"`
		}
		c.ShouldBindJSON(&req)
		
		embedding, err := ollama.Embed(req.Text)
		if err != nil {
			c.JSON(500, gin.H{"error": err.Error()})
			return
		}
		
		c.JSON(200, gin.H{
			"embedding": embedding,
			"dimension": len(embedding),
		})
	})
	
	r.POST("/analyze", func(c *gin.Context) {
		var req struct {
			Code   string   `json:"code"`
			Errors []string `json:"errors"`
		}
		c.ShouldBindJSON(&req)
		
		prompt := fmt.Sprintf("Analyze this code and errors:\n\nCode:\n%s\n\nErrors:\n%v\n\nProvide specific fixes:",
			req.Code, req.Errors)
		
		response, err := ollama.Generate(prompt)
		if err != nil {
			c.JSON(500, gin.H{"error": err.Error()})
			return
		}
		
		c.JSON(200, gin.H{
			"analysis": response,
			"gpu_used": cublasHandle != nil,
		})
	})
}
