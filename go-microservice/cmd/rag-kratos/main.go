package main

// Kratos + Gin + OpenTelemetry service that exposes /embed, /rag, /health, /metrics
// Logs are JSON (stdout) for ELK ingestion; traces/metrics via OTLP and Prometheus.

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strconv"
	"time"

	"github.com/gin-gonic/gin"
	kratos "github.com/go-kratos/kratos/v2"
	klog "github.com/go-kratos/kratos/v2/log"
	khttp "github.com/go-kratos/kratos/v2/transport/http"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"

	otelgin "go.opentelemetry.io/contrib/instrumentation/github.com/gin-gonic/gin/otelgin"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.21.0"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	"crypto/sha256"
	"encoding/hex"
	"sync"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
	pgvector "github.com/pgvector/pgvector-go"
	redis "github.com/redis/go-redis/v9"

	fastjson "legal-ai-production/internal/fastjson"
)

// Types (duplicated minimal structs for independence)
type EmbedRequest struct {
	Texts []string `json:"texts"`
	Model string   `json:"model,omitempty"`
}

type EmbedResponse struct {
	Model   string      `json:"model"`
	Vectors [][]float32 `json:"vectors"`
}

type RAGRequest struct {
	Query string `json:"query"`
	TopK  int    `json:"topK"`
}

type RAGDoc struct {
	ID    string  `json:"id"`
	Text  string  `json:"text"`
	Score float32 `json:"score"`
}

type RAGResponse struct {
	Results []RAGDoc         `json:"results"`
	Graph   *GraphEnrichment `json:"graph,omitempty"`
}

// Graph enrichment types
type GraphNode struct {
	ID    string         `json:"id"`
	Label string         `json:"label"`
	Props map[string]any `json:"props,omitempty"`
}
type GraphEdge struct {
	Source string         `json:"source"`
	Target string         `json:"target"`
	Type   string         `json:"type"`
	Props  map[string]any `json:"props,omitempty"`
}
type GraphEnrichment struct {
	Nodes []GraphNode `json:"nodes"`
	Edges []GraphEdge `json:"edges"`
}

// Minimal clients (same behavior as other cmd, kept local)
type OllamaClient struct {
	baseURL string
	http    *http.Client
}

func NewOllamaClient(base string) *OllamaClient {
	return &OllamaClient{baseURL: base, http: &http.Client{Timeout: 60 * time.Second}}
}
func (o *OllamaClient) EmbedBatch(ctx context.Context, texts []string, model string) ([][]float32, error) {
	out := make([][]float32, 0, len(texts))
	for _, t := range texts {
		req := map[string]any{"model": model, "input": t}
		body, _ := json.Marshal(req)
		httpReq, _ := http.NewRequestWithContext(ctx, http.MethodPost, o.baseURL+"/api/embeddings", bytes.NewReader(body))
		httpReq.Header.Set("Content-Type", "application/json")
		resp, err := o.http.Do(httpReq)
		if err != nil {
			return nil, err
		}
		if resp.StatusCode >= 300 {
			_ = resp.Body.Close()
			return nil, fmt.Errorf("ollama status %d", resp.StatusCode)
		}
		var outOne struct {
			Embedding []float32 `json:"embedding"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&outOne); err != nil {
			_ = resp.Body.Close()
			return nil, err
		}
		_ = resp.Body.Close()
		out = append(out, outOne.Embedding)
	}
	return out, nil
}

// PG stubs; wire pgx + pgvector in follow-up
type PGConfig struct {
	ConnString     string
	Table          string
	IDColumn       string
	TextColumn     string
	VectorColumn   string
	DistanceMetric string // cosine|l2|ip
}
type PGSearch struct {
	cfg  PGConfig
	pool *pgxpool.Pool
}

func NewPGSearch(ctx context.Context, cfg PGConfig) (*PGSearch, error) {
	pool, err := pgxpool.New(ctx, cfg.ConnString)
	if err != nil {
		return nil, err
	}
	// Optionally ensure extension exists (ignore error if insufficient perms)
	_, _ = pool.Exec(ctx, "CREATE EXTENSION IF NOT EXISTS vector")
	if cfg.DistanceMetric == "" {
		cfg.DistanceMetric = env("DISTANCE_METRIC", "cosine")
	}
	return &PGSearch{cfg: cfg, pool: pool}, nil
}
func (p *PGSearch) CosineSearch(ctx context.Context, vec []float32, k int) ([]RAGDoc, error) {
	if p.pool == nil {
		return nil, fmt.Errorf("pg pool not initialized")
	}
	if k <= 0 {
		k = 10
	}
	v := pgvector.NewVector(vec)
	// Build metric-aware SQL
	metric := p.cfg.DistanceMetric
	orderBy := fmt.Sprintf("%s <-> $1", p.cfg.VectorColumn)
	var scoreExpr string
	switch metric {
	case "l2":
		// score ~ [0,1], higher is better
		scoreExpr = "1 / (1 + l2_distance(" + p.cfg.VectorColumn + ", $1)) AS score"
	case "ip":
		// inner_product distance is -dot; convert to similarity in [0,1] approximately via sigmoid
		scoreExpr = "1 / (1 + exp(inner_product(" + p.cfg.VectorColumn + ", $1))) AS score"
	default:
		// cosine (default)
		scoreExpr = "1 - cosine_distance(" + p.cfg.VectorColumn + ", $1) AS score"
	}

	sql := fmt.Sprintf(
		"SELECT %s, %s, %s FROM %s ORDER BY %s LIMIT $2",
		p.cfg.IDColumn,
		p.cfg.TextColumn,
		scoreExpr,
		p.cfg.Table,
		orderBy,
	)

	rows, err := p.pool.Query(ctx, sql, v, k)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := make([]RAGDoc, 0, k)
	for rows.Next() {
		var id, text string
		var score float32
		if err := rows.Scan(&id, &text, &score); err != nil {
			return nil, err
		}
		out = append(out, RAGDoc{ID: id, Text: text, Score: score})
	}
	return out, rows.Err()
}

// -------------------------------
// L1/L2 Embedding Cache
// -------------------------------
type cacheEntry struct {
	vec       []float32
	expiresAt time.Time
}

type EmbedCache struct {
	mu   sync.RWMutex
	ttl  time.Duration
	l1   map[string]cacheEntry
	rdb  *redis.Client
	pref string
}

func NewEmbedCacheFromEnv() *EmbedCache {
	ttlStr := env("EMBED_CACHE_TTL", "10m")
	ttl, err := time.ParseDuration(ttlStr)
	if err != nil {
		ttl = 10 * time.Minute
	}
	c := &EmbedCache{ttl: ttl, l1: make(map[string]cacheEntry), pref: "emb:"}
	if addr := os.Getenv("REDIS_ADDR"); addr != "" {
		db := 0
		if v := os.Getenv("REDIS_DB"); v != "" {
			if n, err := strconv.Atoi(v); err == nil {
				db = n
			}
		}
		c.rdb = redis.NewClient(&redis.Options{
			Addr:     addr,
			Password: os.Getenv("REDIS_PASSWORD"),
			DB:       db,
		})
	}
	return c
}

func (c *EmbedCache) key(model, text string) string {
	sum := sha256.Sum256([]byte(model + "|" + text))
	return c.pref + model + ":" + hex.EncodeToString(sum[:])
}

func (c *EmbedCache) Get(ctx context.Context, model, text string) ([]float32, bool) {
	k := c.key(model, text)
	// L1
	c.mu.RLock()
	if ent, ok := c.l1[k]; ok {
		if time.Now().Before(ent.expiresAt) {
			c.mu.RUnlock()
			return ent.vec, true
		}
	}
	c.mu.RUnlock()
	// L2
	if c.rdb != nil {
		if s, err := c.rdb.Get(ctx, k).Result(); err == nil && s != "" {
			var v []float32
			if err := json.Unmarshal([]byte(s), &v); err == nil && len(v) > 0 {
				// populate L1
				c.mu.Lock()
				c.l1[k] = cacheEntry{vec: v, expiresAt: time.Now().Add(c.ttl)}
				c.mu.Unlock()
				return v, true
			}
		}
	}
	return nil, false
}

func (c *EmbedCache) Set(ctx context.Context, model, text string, vec []float32) {
	k := c.key(model, text)
	// L1
	c.mu.Lock()
	c.l1[k] = cacheEntry{vec: vec, expiresAt: time.Now().Add(c.ttl)}
	c.mu.Unlock()
	// L2
	if c.rdb != nil {
		b, _ := json.Marshal(vec)
		_ = c.rdb.Set(ctx, k, string(b), c.ttl).Err()
	}
}

var (
	embedCacheHits   = prometheus.NewCounter(prometheus.CounterOpts{Name: "embed_cache_hits_total", Help: "Total embedding cache hits"})
	embedCacheMisses = prometheus.NewCounter(prometheus.CounterOpts{Name: "embed_cache_misses_total", Help: "Total embedding cache misses"})
)

// Env/config
func env(k, d string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return d
}

// Telemetry setup
func setupTracer(ctx context.Context, svcName, svcVersion string) (*sdktrace.TracerProvider, error) {
	endpoint := env("OTEL_EXPORTER_OTLP_ENDPOINT", "localhost:4317")
	exp, err := otlptracegrpc.New(ctx,
		otlptracegrpc.WithEndpoint(endpoint),
		otlptracegrpc.WithDialOption(grpc.WithTransportCredentials(insecure.NewCredentials())),
	)
	if err != nil {
		return nil, err
	}
	res, _ := resource.Merge(resource.Default(), resource.NewWithAttributes(
		semconv.SchemaURL,
		semconv.ServiceName(svcName),
		semconv.ServiceVersion(svcVersion),
		attribute.String("env", env("ENV", "dev")),
	))
	tp := sdktrace.NewTracerProvider(
		sdktrace.WithBatcher(exp),
		sdktrace.WithResource(res),
	)
	otel.SetTracerProvider(tp)
	otel.SetTextMapPropagator(propagation.TraceContext{})
	return tp, nil
}

func main() {
	// Logger (JSON to stdout)
	logger := klog.With(klog.NewStdLogger(os.Stdout),
		"ts", klog.DefaultTimestamp,
		"caller", klog.DefaultCaller,
		"service", "rag-kratos",
	)
	helper := klog.NewHelper(logger)

	// Telemetry
	ctx := context.Background()
	tp, err := setupTracer(ctx, "rag-kratos", env("SERVICE_VERSION", "0.1.0"))
	if err != nil {
		helper.Errorf("otel setup: %v", err)
	}
	defer func() {
		if tp != nil {
			_ = tp.Shutdown(context.Background())
		}
	}()

	// Dependencies
	var ollama *OllamaClient
	if base := env("OLLAMA_BASE_URL", "http://localhost:11434"); base != "" {
		ollama = NewOllamaClient(base)
	}
	// Cache and metrics
	prometheus.MustRegister(embedCacheHits, embedCacheMisses)
	cache := NewEmbedCacheFromEnv()
	var pg *PGSearch
	if dsn := os.Getenv("PG_CONN_STRING"); dsn != "" {
		cfg := PGConfig{ConnString: dsn, Table: env("VECTOR_TABLE", "documents"), IDColumn: env("ID_COLUMN", "id"), TextColumn: env("TEXT_COLUMN", "text"), VectorColumn: env("VECTOR_COLUMN", "embedding"), DistanceMetric: env("DISTANCE_METRIC", "cosine")}
		if s, err := NewPGSearch(ctx, cfg); err == nil {
			pg = s
		} else {
			helper.Warnf("pg init: %v", err)
		}
	}
	embedModel := env("EMBED_MODEL", "nomic-embed-text")

	// Optional Neo4j
	var neo neo4j.DriverWithContext
	if uri := os.Getenv("NEO4J_URI"); uri != "" {
		user := os.Getenv("NEO4J_USER")
		pass := os.Getenv("NEO4J_PASSWORD")
		drv, err := neo4j.NewDriverWithContext(uri, neo4j.BasicAuth(user, pass, ""))
		if err == nil {
			neo = drv
			defer neo.Close(context.Background())
		} else {
			helper.Warnf("neo4j init: %v", err)
		}
	}

	// Gin engine with OTel
	gin.SetMode(gin.ReleaseMode)
	r := gin.New()
	r.Use(gin.Recovery())
	r.Use(otelgin.Middleware("rag-kratos"))

	// Routes
	r.GET("/health", func(c *gin.Context) { c.JSON(200, gin.H{"status": "ok", "time": time.Now().Format(time.RFC3339)}) })
	r.GET("/metrics", gin.WrapH(promhttp.Handler()))

		r.POST("/embed", func(c *gin.Context) {
		stream := c.Query("stream") == "1"
		var req EmbedRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(400, gin.H{"error": err.Error()})
			return
		}
		model := req.Model
		if model == "" {
			model = embedModel
		}
		// Handle trivial cases
		if len(req.Texts) == 0 {
			c.JSON(200, EmbedResponse{Model: model, Vectors: [][]float32{}})
			return
		}
		// Attempt cache per text
		result := make([][]float32, len(req.Texts))
		missingIdx := make([]int, 0)
		missingTexts := make([]string, 0)
		for i, t := range req.Texts {
			if v, ok := cache.Get(c.Request.Context(), model, t); ok {
				result[i] = v
				embedCacheHits.Inc()
			} else {
				missingIdx = append(missingIdx, i)
				missingTexts = append(missingTexts, t)
				embedCacheMisses.Inc()
			}
		}
		// If any missing and ollama available, fetch and fill
		if len(missingTexts) > 0 {
			if ollama == nil {
				c.JSON(200, EmbedResponse{Model: model, Vectors: result})
				return
			}
			vecs, err := ollama.EmbedBatch(c.Request.Context(), missingTexts, model)
			if err != nil {
				c.JSON(502, gin.H{"error": err.Error()})
				return
			}
			for j, idx := range missingIdx {
				if idx >= 0 && idx < len(result) && j < len(vecs) {
					result[idx] = vecs[j]
					cache.Set(c.Request.Context(), model, req.Texts[idx], vecs[j])
				}
			}
		}
		if !stream {
			if b, err := fastjson.Marshal(EmbedResponse{Model: model, Vectors: result}); err == nil {
				c.Writer.Header().Set("Content-Type", "application/json")
				c.Writer.WriteHeader(200)
				c.Writer.Write(b)
				return
			}
			c.JSON(200, EmbedResponse{Model: model, Vectors: result})
			return
		}
		// Streaming one vector per line
		w := c.Writer
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Transfer-Encoding", "chunked")
		flusher, ok := w.(http.Flusher)
		if !ok { c.JSON(500, gin.H{"error": "stream unsupported"}); return }
		meta := map[string]any{"model": model, "count": len(result), "stream": true, "ts": time.Now().Format(time.RFC3339)}
		b, _ := json.Marshal(meta)
		w.Write(b); w.Write([]byte("\n")); flusher.Flush()
		for i, v := range result {
			line := map[string]any{"index": i, "vector": v}
			lb, _ := json.Marshal(line)
			w.Write(lb); w.Write([]byte("\n")); flusher.Flush()
		}
		final := map[string]any{"complete": true, "total": len(result)}
		fb, _ := json.Marshal(final)
		w.Write(fb)
		flusher.Flush()
	})

	r.POST("/rag", func(c *gin.Context) {
		var req RAGRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(400, gin.H{"error": err.Error()})
			return
		}
		if req.TopK <= 0 {
			req.TopK = 10
		}
		stream := c.Query("stream") == "1"
		// Embed query
		var qvec []float32
		if ollama != nil {
			// Try cache first
			if v, ok := cache.Get(c.Request.Context(), embedModel, req.Query); ok {
				qvec = v
				embedCacheHits.Inc()
			} else {
				v2, err := ollama.EmbedBatch(c.Request.Context(), []string{req.Query}, embedModel)
				if err != nil || len(v2) == 0 {
					c.JSON(502, gin.H{"error": fmt.Sprintf("embed failed: %v", err)})
					return
				}
				qvec = v2[0]
				cache.Set(c.Request.Context(), embedModel, req.Query, qvec)
			}
		} else {
			qvec = make([]float32, 128)
		}
		if pg == nil {
			c.JSON(200, RAGResponse{Results: []RAGDoc{{ID: "demo", Text: "stub", Score: 0.9}}})
			return
		}
		res, err := pg.CosineSearch(c.Request.Context(), qvec, req.TopK)
		if err != nil {
			c.JSON(502, gin.H{"error": err.Error()})
			return
		}

		if stream {
			w := c.Writer
			w.Header().Set("Content-Type", "application/x-ndjson; charset=utf-8")
			w.Header().Set("Cache-Control", "no-cache")
			w.Header().Set("Transfer-Encoding", "chunked")
			flusher, ok := w.(http.Flusher)
			if !ok { c.JSON(500, gin.H{"error": "stream unsupported"}); return }
			// Send metadata line
			meta := map[string]any{"query": req.Query, "topK": req.TopK, "count": len(res), "stream": true, "ts": time.Now().Format(time.RFC3339)}
			mb, _ := json.Marshal(meta)
			w.Write(mb); w.Write([]byte("\n")); flusher.Flush()
			for i, doc := range res {
				line := map[string]any{"index": i, "id": doc.ID, "score": doc.Score, "text": doc.Text}
				b, _ := json.Marshal(line)
				w.Write(b); w.Write([]byte("\n")); flusher.Flush()
			}
			final := map[string]any{"complete": true, "total": len(res)}
			fb, _ := json.Marshal(final)
			w.Write(fb)
			flusher.Flush()
			return
		}
		// Optional graph enrichment if ?graph=1 and neo4j configured
		if c.Query("graph") == "1" && neo != nil {
			graph := &GraphEnrichment{Nodes: make([]GraphNode, 0), Edges: make([]GraphEdge, 0)}
			// Dedup via sets
			nodeSeen := make(map[string]bool)
			edgeSeen := make(map[string]bool)
			sess := neo.NewSession(context.Background(), neo4j.SessionConfig{AccessMode: neo4j.AccessModeRead})
			defer sess.Close(context.Background())
			// Cap total traversals
			maxPerDoc := 50
			// Query with elementId for stable identifiers
			cypher := `MATCH (d:Document {id:$id})-[r]-(n)
RETURN elementId(d) as did, labels(n) as labels, properties(n) as props, type(r) as relType, elementId(startNode(r)) as src, elementId(endNode(r)) as tgt
LIMIT $lim`

			for _, d := range res {
				// Add the document node once
				if !nodeSeen[d.ID] {
					graph.Nodes = append(graph.Nodes, GraphNode{ID: d.ID, Label: "Document"})
					nodeSeen[d.ID] = true
				}
				// Use bounded context for safety
				gctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
				_, _ = sess.ExecuteRead(gctx, func(tx neo4j.ManagedTransaction) (any, error) {
					rows, err := tx.Run(gctx, cypher, map[string]any{"id": d.ID, "lim": maxPerDoc})
					if err != nil {
						return nil, err
					}
					for rows.Next(gctx) {
						rec := rows.Record()
						labelsAny, _ := rec.Get("labels")
						propsAny, _ := rec.Get("props")
						relTypeAny, _ := rec.Get("relType")
						srcAny, _ := rec.Get("src")
						tgtAny, _ := rec.Get("tgt")
						lbls, _ := labelsAny.([]any)
						label := ""
						if len(lbls) > 0 {
							if s, ok := lbls[0].(string); ok {
								label = s
							}
						}
						props, _ := propsAny.(map[string]any)
						// Prefer node's own id property; fallback to elementId
						nid := ""
						if props != nil {
							if v, ok := props["id"]; ok {
								nid = fmt.Sprintf("%v", v)
							}
						}
						if nid == "" {
							nid = fmt.Sprintf("%v", tgtAny)
						}
						if !nodeSeen[nid] {
							graph.Nodes = append(graph.Nodes, GraphNode{ID: nid, Label: label, Props: props})
							nodeSeen[nid] = true
						}
						// Edge key
						ekey := fmt.Sprintf("%v|%v|%v", srcAny, tgtAny, relTypeAny)
						if !edgeSeen[ekey] {
							graph.Edges = append(graph.Edges, GraphEdge{Source: fmt.Sprintf("%v", srcAny), Target: fmt.Sprintf("%v", tgtAny), Type: fmt.Sprintf("%v", relTypeAny)})
							edgeSeen[ekey] = true
						}
					}
					return nil, rows.Err()
				})
				cancel()
			}
			c.JSON(200, RAGResponse{Results: res, Graph: graph})
			return
		}
		c.JSON(200, RAGResponse{Results: res})
	})

	// Kratos HTTP server wrapping gin
	addr := ":" + env("RAG_HTTP_PORT", "8093")
	srv := khttp.NewServer(khttp.Address(addr))
	srv.HandlePrefix("/", r)

	app := kratos.New(
		kratos.Name("rag-kratos"),
		kratos.Version(env("SERVICE_VERSION", "0.1.0")),
		kratos.Logger(logger),
		kratos.Server(srv),
	)
	helper.Infof("starting rag-kratos on %s (otel=%v, pg=%v)", addr, tp != nil, pg != nil)
	if err := app.Run(); err != nil {
		helper.Errorf("app run: %v", err)
	}
}
