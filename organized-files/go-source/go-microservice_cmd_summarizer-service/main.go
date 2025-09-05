package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"os"
	"strconv"
	"sync"
	"time"

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

func makeCacheKey(model, format string, maxTok int, text string) string {
	h := sha256.New()
	h.Write([]byte(model))
	h.Write([]byte{'|'})
	h.Write([]byte(format))
	h.Write([]byte{'|'})
	h.Write([]byte(strconv.Itoa(maxTok)))
	h.Write([]byte{'|'})
	h.Write([]byte(text))
	return "summarize:" + hex.EncodeToString(h.Sum(nil))
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
	acquireTimeoutMs := 1500
	if v := os.Getenv("SUMMARIZER_ACQUIRE_TIMEOUT_MS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			acquireTimeoutMs = n
		}
	}

	r := gin.Default()

	// metrics state
	m := &metrics{StartTime: time.Now().Unix()}

	// WebSocket/SSE live agent routes are available in dev servers; not mounted here

	// caches
	// L1 in-process (defaults)
	l1TTL := 60
	if v := os.Getenv("SUMMARIZER_L1_TTL_SEC"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			l1TTL = n
		}
	}
	l1Max := 512
	if v := os.Getenv("SUMMARIZER_L1_MAX"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			l1Max = n
		}
	}
	l1 := newLocalCache(time.Duration(l1TTL)*time.Second, l1Max)

	// L2 Redis (optional)
	var rdb *redis.Client
	var cacheTTL time.Duration
	if addr := os.Getenv("REDIS_ADDR"); addr != "" {
		rdb = redis.NewClient(&redis.Options{Addr: addr})
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

	r.GET("/metrics", func(c *gin.Context) {
		m.mu.Lock()
		m.UptimeSec = time.Now().Unix() - m.StartTime
		snap := struct {
			Total         int64   `json:"total"`
			Success       int64   `json:"success"`
			Errors        int64   `json:"errors"`
			CacheHits     int64   `json:"cacheHits"`
			CacheMisses   int64   `json:"cacheMisses"`
			AvgLatencyMs  float64 `json:"avgLatencyMs"`
			LastLatencyMs int64   `json:"lastLatencyMs"`
			UptimeSec     int64   `json:"uptimeSec"`
			StartTime     int64   `json:"startTime"`
		}{
			Total: m.Total, Success: m.Success, Errors: m.Errors,
			CacheHits: m.CacheHits, CacheMisses: m.CacheMisses,
			AvgLatencyMs: m.AvgLatencyMs, LastLatencyMs: m.LastLatencyMs,
			UptimeSec: m.UptimeSec, StartTime: m.StartTime,
		}
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
		// backpressure-aware acquire
		if !tryAcquire(sem, time.Duration(acquireTimeoutMs)*time.Millisecond) {
			c.Header("Retry-After", "1")
			c.JSON(429, gin.H{"error": "busy, try again"})
			return
		}
		defer func() { <-sem }()
		var req summarizeReq
		if err := c.BindJSON(&req); err != nil {
			m.observe(0, false)
			c.JSON(400, gin.H{"error": "invalid json"})
			return
		}
		if len(req.Text) == 0 {
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
		cacheKey := makeCacheKey(model, format, maxTok, req.Text)

		// L1
		if data, ok := l1.get(cacheKey); ok {
			m.mu.Lock()
			m.CacheHits++
			m.mu.Unlock()
			c.Header("ETag", cacheKey)
			c.Header("X-Cache", "L1-HIT")
			c.Header("Cache-Control", "public, max-age=60")
			c.Data(200, "application/json", data)
			return
		}
		// L2
		if rdb != nil {
			ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
			if val, err := rdb.Get(ctx, cacheKey).Result(); err == nil && val != "" {
				cancel()
				m.mu.Lock()
				m.CacheHits++
				m.mu.Unlock()
				l1.set(cacheKey, []byte(val))
				c.Header("ETag", cacheKey)
				c.Header("X-Cache", "L2-REDIS-HIT")
				c.Header("Cache-Control", "public, max-age=60")
				c.Data(200, "application/json", []byte(val))
				return
			}
			cancel()
			m.mu.Lock()
			m.CacheMisses++
			m.mu.Unlock()
		}

		prompt := "Summarize the following text concisely. If format is 'bullets', return 3-7 bullet points. If 'summary', return 1 concise paragraph.\nFormat: " + format + "\n---\n" + req.Text

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
		m.observe(time.Since(start), true)
		respJSON, _ := json.Marshal(gin.H{
			"model":    model,
			"format":   format,
			"response": ogResp.Response,
			"done":     ogResp.Done,
		})
		// write-through caches
		l1.set(cacheKey, respJSON)
		if rdb != nil {
			ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
			_ = rdb.Set(ctx, cacheKey, string(respJSON), cacheTTL).Err()
			cancel()
		}
		c.Header("ETag", cacheKey)
		c.Header("X-Cache", "MISS-GEN")
		c.Header("Cache-Control", "public, max-age=60")
		c.Data(200, "application/json", respJSON)
	})

	// Streaming summarization via SSE (proxies Ollama's JSONL stream)
	r.POST("/summarize/stream", func(c *gin.Context) {
		if !tryAcquire(sem, time.Duration(acquireTimeoutMs)*time.Millisecond) {
			c.Header("Retry-After", "1")
			c.JSON(429, gin.H{"error": "busy, try again"})
			return
		}
		defer func() { <-sem }()

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
		}
		prompt := "Summarize the following text concisely. If format is 'bullets', return 3-7 bullet points. If 'summary', return 1 concise paragraph.\nFormat: " + format + "\n---\n" + req.Text

		og := ollamaGenReq{Model: model, Prompt: prompt, Stream: true, Options: map[string]interface{}{"num_predict": maxTok}}
		b, _ := json.Marshal(og)
		client := &http.Client{Timeout: 0}
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
			resp.Body.Close()
			m.observe(time.Since(start), false)
			c.JSON(500, gin.H{"error": "streaming unsupported"})
			return
		}
		scanner := bufio.NewScanner(resp.Body)
		buf := make([]byte, 0, 64*1024)
		scanner.Buffer(buf, 1024*1024)
		for scanner.Scan() {
			line := scanner.Bytes()
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
