package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"os"
	"strconv"
	"sync"
	"time"
	"context"
	"crypto/sha256"
	"encoding/hex"

	"github.com/gin-gonic/gin"
	redis "github.com/redis/go-redis/v9"
)

type healthResp struct {
	OK              bool   `json:"ok"`
	Port            string `json:"port"`
	OllamaBaseURL   string `json:"ollamaBaseUrl"`
	Model           string `json:"model"`
	OllamaReachable bool   `json:"ollamaReachable"`
	ModelAvailable  bool   `json:"modelAvailable"`
	Message         string `json:"message,omitempty"`
}

type summarizeReq struct {
	Text      string `json:"text"`
	MaxTokens int    `json:"maxTokens"`
	Model     string `json:"model"`
	Format    string `json:"format"` // "bullets" | "summary"
}

type ollamaTagsResp struct {
	Models []struct {
		Name string `json:"name"`
	} `json:"models"`
}

type ollamaGenReq struct {
	Model   string                 `json:"model"`
	Prompt  string                 `json:"prompt"`
	Stream  bool                   `json:"stream"`
	Options map[string]interface{} `json:"options,omitempty"`
}

type ollamaGenResp struct {
	Response string `json:"response"`
	Done     bool   `json:"done"`
}

// simple in-memory metrics
type metrics struct {
	mu            sync.Mutex
	Total         int64   `json:"total"`
	Success       int64   `json:"success"`
	Errors        int64   `json:"errors"`
	CacheHits     int64   `json:"cacheHits"`
	CacheMisses   int64   `json:"cacheMisses"`
	AvgLatencyMs  float64 `json:"avgLatencyMs"`
	LastLatencyMs int64   `json:"lastLatencyMs"`
	UptimeSec     int64   `json:"uptimeSec"`
	StartTime     int64   `json:"startTime"`
}

func (m *metrics) observe(latency time.Duration, ok bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.Total++
	if ok {
		m.Success++
	} else {
		m.Errors++
	}
	m.LastLatencyMs = latency.Milliseconds()
	// simple EMA for avg latency
	if m.AvgLatencyMs == 0 {
		m.AvgLatencyMs = float64(m.LastLatencyMs)
	} else {
		m.AvgLatencyMs = 0.9*m.AvgLatencyMs + 0.1*float64(m.LastLatencyMs)
	}
}

// simple in-process TTL cache (L1)
type cacheEntry struct {
	data   []byte
	expire time.Time
}

type localCache struct {
	mu    sync.Mutex
	items map[string]cacheEntry
	ttl   time.Duration
	max   int
}

func newLocalCache(ttl time.Duration, max int) *localCache {
	return &localCache{items: make(map[string]cacheEntry), ttl: ttl, max: max}
}

func (lc *localCache) get(key string) ([]byte, bool) {
	lc.mu.Lock()
	defer lc.mu.Unlock()
	if e, ok := lc.items[key]; ok {
		if time.Now().Before(e.expire) {
			return e.data, true
		}
		delete(lc.items, key)
	}
	return nil, false
}

func (lc *localCache) set(key string, val []byte) {
	lc.mu.Lock()
	defer lc.mu.Unlock()
	if lc.max > 0 && len(lc.items) >= lc.max {
		for k := range lc.items {
			delete(lc.items, k)
			break
		}
	}
	lc.items[key] = cacheEntry{data: val, expire: time.Now().Add(lc.ttl)}
}

func tryAcquire(sem chan struct{}, timeout time.Duration) bool {
	select {
	case sem <- struct{}{}:
		return true
	case <-time.After(timeout):
		return false
	}
}

func getEnv(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

func checkOllama(baseURL string, model string) (reachable bool, modelAvailable bool) {
	client := &http.Client{Timeout: 2 * time.Second}
	// reachability
	resp, err := client.Get(baseURL + "/api/version")
	if err == nil {
		reachable = resp.StatusCode == 200
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
	}
	if !reachable {
		return false, false
	}
	// model availability via /api/tags
	tagsResp, err := client.Get(baseURL + "/api/tags")
	if err != nil {
		return true, false
	}
	defer tagsResp.Body.Close()
	var tags ollamaTagsResp
	if err := json.NewDecoder(tagsResp.Body).Decode(&tags); err != nil {
		return true, false
	}
	for _, m := range tags.Models {
		if m.Name == model {
			return true, true
		}
	}
	return true, false
}

func main() {
	port := getEnv("SUMMARIZER_HTTP_PORT", "8091")
	ollamaBase := getEnv("OLLAMA_BASE_URL", "http://localhost:11434")
	defaultModel := getEnv("OLLAMA_MODEL", "llama3.1:8b")
	// concurrency limit
	maxConc := 2
	if v := os.Getenv("SUMMARIZER_MAX_CONCURRENCY"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			maxConc = n
		}
	}
	sem := make(chan struct{}, maxConc)

	r := gin.Default()

	// metrics state
	m := &metrics{StartTime: time.Now().Unix()}

	r.GET("/metrics", func(c *gin.Context) {
		m.mu.Lock()
		m.UptimeSec = time.Now().Unix() - m.StartTime
		snap := *m
		m.mu.Unlock()
		c.JSON(200, snap)
	})

	r.GET("/health", func(c *gin.Context) {
		reach, avail := checkOllama(ollamaBase, defaultModel)
		c.JSON(200, healthResp{
			OK:              true,
			Port:            port,
			OllamaBaseURL:   ollamaBase,
			Model:           defaultModel,
			OllamaReachable: reach,
			ModelAvailable:  avail,
		})
	})

	r.POST("/summarize", func(c *gin.Context) {
		// acquire concurrency slot
		startTotal := time.Now()
		sem <- struct{}{}
		defer func() { <-sem }()
		var req summarizeReq
		if err := c.BindJSON(&req); err != nil {
			m.observe(0, false)
			c.JSON(400, gin.H{"error": "invalid json"})
			return
		}
		text := req.Text
		if len(text) == 0 {
			m.observe(0, false)
			c.JSON(400, gin.H{"error": "text is required"})
			return
		}
		model := req.Model
		if model == "" {
			model = defaultModel
		}
		// Avoid triggering downloads: ensure model is available first
		reach, avail := checkOllama(ollamaBase, model)
		if !reach {
			m.observe(0, false)
			c.JSON(502, gin.H{"error": "ollama not reachable", "ollamaBaseUrl": ollamaBase})
				// optional Redis cache
				var rdb *redis.Client
				var cacheTTL time.Duration
				if addr := os.Getenv("REDIS_ADDR"); addr != "" {
					rdb = redis.NewClient(&redis.Options{Addr: addr})
					// quick ping with short timeout
					ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
					if err := rdb.Ping(ctx).Err(); err != nil {
						log.Printf("[summarizer] redis disabled (ping failed): %v", err)
						rdb = nil
					}
					cancel()
					ttlSec := 600
					if v := os.Getenv("SUMMARIZER_CACHE_TTL_SEC"); v != "" {
						if n, err := strconv.Atoi(v); err == nil && n > 0 {
							ttlSec = n
						}
					}
					cacheTTL = time.Duration(ttlSec) * time.Second
					if rdb != nil {
						log.Printf("[summarizer] redis cache enabled at %s (ttl %ds)", addr, ttlSec)
					}
				}
			return
		}
		if !avail {
			m.observe(0, false)
			c.JSON(412, gin.H{"error": "model not available locally; please pull manually to avoid downloads", "model": model})
			return
		}

		maxTok := req.MaxTokens
		if maxTok <= 0 {
			maxTok = 256
		}
		format := req.Format
		if format == "" {
			format = "bullets"
		}
		prompt := "Summarize the following text concisely. If format is 'bullets', return 3-7 bullet points. If 'summary', return 1 concise paragraph.\nFormat: " + format + "\n---\n" + text

		og := ollamaGenReq{
			Model:  model,
			Prompt: prompt,
			Stream: false,
			Options: map[string]interface{}{
				"num_predict": maxTok,
			},
		}
		b, _ := json.Marshal(og)
		client := &http.Client{Timeout: 60 * time.Second}
		start := time.Now()
		resp, err := client.Post(ollamaBase+"/api/generate", "application/json", bytes.NewReader(b))
		if err != nil {
			m.observe(time.Since(start), false)
			c.JSON(502, gin.H{"error": err.Error()})
			return
		}
		defer resp.Body.Close()
		if resp.StatusCode >= 400 {
			body, _ := io.ReadAll(resp.Body)
			m.observe(time.Since(start), false)
			c.JSON(502, gin.H{"error": "ollama error", "status": resp.StatusCode, "body": string(body)})
			return
		}
		var ogResp ollamaGenResp
		if err := json.NewDecoder(resp.Body).Decode(&ogResp); err != nil {
			m.observe(time.Since(start), false)
			c.JSON(502, gin.H{"error": "invalid ollama response"})
			return
		}
		_ = startTotal // reserved for potential end-to-end metric
		m.observe(time.Since(start), true)
		c.JSON(200, gin.H{
			"model":    model,
			"format":   format,
			"response": ogResp.Response,
			"done":     ogResp.Done,
		})
	})

	// Streaming summarization via SSE (proxies Ollama's JSONL stream)
	r.POST("/summarize/stream", func(c *gin.Context) {
		// acquire concurrency slot
		sem <- struct{}{}
		defer func() { <-sem }()

					// Cache key on model+format+maxTok+text
					var cacheKey string
					if rdb != nil {
						h := sha256.New()
						h.Write([]byte(model))
						h.Write([]byte{"|"[0]})
						h.Write([]byte(format))
						h.Write([]byte{"|"[0]})
						h.Write([]byte(strconv.Itoa(maxTok)))
						h.Write([]byte{"|"[0]})
						h.Write([]byte(text))
						cacheKey = "summarize:" + hex.EncodeToString(h.Sum(nil))
						// Try cache read
						ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
						if val, err := rdb.Get(ctx, cacheKey).Result(); err == nil && val != "" {
							cancel()
							m.mu.Lock(); m.CacheHits++; m.mu.Unlock()
							// Return cached JSON directly
							c.Data(200, "application/json", []byte(val))
							return
						} else {
							cancel()
							m.mu.Lock(); m.CacheMisses++; m.mu.Unlock()
						}
					}

		var req summarizeReq
		if err := c.BindJSON(&req); err != nil {
			m.observe(0, false)
			c.JSON(400, gin.H{"error": "invalid json"})
			return
		}
		if req.Text == "" {
			m.observe(0, false)
			c.JSON(400, gin.H{"error": "text is required"})
			return
		}
		model := req.Model
		if model == "" {
			model = defaultModel
		}
		reach, avail := checkOllama(ollamaBase, model)
		if !reach {
			m.observe(0, false)
			c.JSON(502, gin.H{"error": "ollama not reachable", "ollamaBaseUrl": ollamaBase})
			return
		}
		if !avail {
			m.observe(0, false)
			c.JSON(412, gin.H{"error": "model not available locally; please pull manually to avoid downloads", "model": model})
			return
		}

		maxTok := req.MaxTokens
		if maxTok <= 0 {
			maxTok = 256
		}
		format := req.Format
		if format == "" {
			format = "bullets"
					// Assemble final JSON once to support caching
					respJSON, _ := json.Marshal(gin.H{
						"model":    model,
						"format":   format,
						"response": ogResp.Response,
						"done":     ogResp.Done,
					})
					// Write-through cache
					if rdb != nil {
						ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
						_ = rdb.Set(ctx, cacheKey, string(respJSON), cacheTTL).Err()
						cancel()
					}
					c.Data(200, "application/json", respJSON)
		}
		prompt := "Summarize the following text concisely. If format is 'bullets', return 3-7 bullet points. If 'summary', return 1 concise paragraph.\nFormat: " + format + "\n---\n" + req.Text

		og := ollamaGenReq{
			Model:  model,
			Prompt: prompt,
			Stream: true,
			Options: map[string]interface{}{
				"num_predict": maxTok,
			},
		}
		b, _ := json.Marshal(og)
		client := &http.Client{Timeout: 0} // no timeout for streams
		start := time.Now()
		resp, err := client.Post(ollamaBase+"/api/generate", "application/json", bytes.NewReader(b))
		if err != nil {
			m.observe(time.Since(start), false)
			c.JSON(502, gin.H{"error": err.Error()})
			return
		}

		if resp.StatusCode >= 400 {
			body, _ := io.ReadAll(resp.Body)
			resp.Body.Close()
			m.observe(time.Since(start), false)
			c.JSON(502, gin.H{"error": "ollama error", "status": resp.StatusCode, "body": string(body)})
			return
		}

		// stream back as Server-Sent Events with JSON payloads per line
		c.Header("Content-Type", "text/event-stream")
		c.Header("Cache-Control", "no-cache")
		c.Header("Connection", "keep-alive")
		c.Header("X-Accel-Buffering", "no")

		writer := c.Writer
		flusher, ok := writer.(http.Flusher)
		if !ok {
			// unlikely with Gin/Go HTTP
			resp.Body.Close()
			m.observe(time.Since(start), false)
			c.JSON(500, gin.H{"error": "streaming unsupported"})
			return
		}

		scanner := bufio.NewScanner(resp.Body)
		// increase buffer for long lines
		buf := make([]byte, 0, 64*1024)
		scanner.Buffer(buf, 1024*1024)
		for scanner.Scan() {
			line := scanner.Bytes()
			// write SSE frame
			writer.Write([]byte("data: "))
			writer.Write(line)
			writer.Write([]byte("\n\n"))
			flusher.Flush()
		}
		resp.Body.Close()
		if err := scanner.Err(); err != nil {
			m.observe(time.Since(start), false)
			return
		}
		m.observe(time.Since(start), true)
	})

	addr := ":" + port
	log.Println("Summarizer service listening on", addr)
	if err := r.Run(addr); err != nil {
		log.Fatal(err)
	}
}
