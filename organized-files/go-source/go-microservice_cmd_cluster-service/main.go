package main

import (
	"log"
	"math/rand"
	"net/http"
	"os"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"

	"microservice/clustering"
)

// request/response DTOs
type ClusterRequest struct {
	Algorithm string            `json:"algorithm"`
	Params    clustering.Params `json:"params"`
	Data      [][]float64       `json:"data"`
}
type ClusterResponse struct {
	JobID       string      `json:"jobId"`
	Algorithm   string      `json:"algorithm"`
	Assignments []int       `json:"assignments"`
	Centroids   [][]float64 `json:"centroids"`
	DurationMs  int64       `json:"durationMs"`
	Error       string      `json:"error,omitempty"`
}

type metrics struct {
	mu            sync.Mutex
	Total         int64   `json:"total"`
	Success       int64   `json:"success"`
	Errors        int64   `json:"errors"`
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
	if m.AvgLatencyMs == 0 {
		m.AvgLatencyMs = float64(m.LastLatencyMs)
	} else {
		m.AvgLatencyMs = 0.9*m.AvgLatencyMs + 0.1*float64(m.LastLatencyMs)
	}
}

func main() {
	// gin in release by default unless DEBUG
	if os.Getenv("GIN_MODE") == "" {
		gin.SetMode(gin.ReleaseMode)
	}
	r := gin.Default()

	// metrics state
	m := &metrics{StartTime: time.Now().Unix()}

	// available algorithms (CPU by default; GPU stub path logs and uses CPU until enabled)
	algos := map[string]clustering.Algorithm{
		"kmeans": clustering.KMeansCPU{},
	}

	if os.Getenv("CLUSTER_MODE") == "gpu" {
		// No downloads or extra deps here — just acknowledge and keep the same implementation
		// Swap-in GPU implementation later behind the same interface
		log.Println("[cluster] GPU mode requested; using CPU implementation until GPU module is enabled")
	}

	r.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"status":     "ok",
			"algorithms": []string{"kmeans"},
		})
	})

	r.GET("/metrics", func(c *gin.Context) {
		m.mu.Lock()
		m.UptimeSec = time.Now().Unix() - m.StartTime
		snap := *m
		m.mu.Unlock()
		c.JSON(http.StatusOK, snap)
	})

	r.POST("/cluster", func(c *gin.Context) {
		var req ClusterRequest
		if err := c.BindJSON(&req); err != nil {
			m.observe(0, false)
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		if req.Algorithm == "" {
			req.Algorithm = "kmeans"
		}
		algo, ok := algos[req.Algorithm]
		if !ok {
			c.JSON(http.StatusBadRequest, gin.H{"error": "unknown algorithm"})
			return
		}
		// default k to min(4, n)
		if req.Params.K <= 0 {
			if n := len(req.Data); n > 0 {
				if n < 4 {
					req.Params.K = n
				} else {
					req.Params.K = 4
				}
			} else {
				req.Params.K = 1
			}
		}
		// optional seed for reproducibility
		if req.Params.Seed != 0 {
			rand.Seed(req.Params.Seed)
		}

		start := time.Now()
		assignments, centroids, err := algo.Cluster(req.Data, req.Params)
		dur := time.Since(start).Milliseconds()
		resp := ClusterResponse{
			JobID:       uuid.NewString(),
			Algorithm:   algo.Name(),
			Assignments: assignments,
			Centroids:   centroids,
			DurationMs:  dur,
		}
		if err != nil {
			resp.Error = err.Error()
			m.observe(time.Since(start), false)
			c.JSON(http.StatusBadRequest, resp)
			return
		}
		m.observe(time.Since(start), true)
		c.JSON(http.StatusOK, resp)
	})

	port := os.Getenv("CLUSTER_HTTP_PORT")
	if port == "" {
		port = "8085"
	}
	addr := ":" + port
	log.Printf("Cluster service listening on %s", addr)
	if err := r.Run(addr); err != nil {
		log.Fatalf("cluster service error: %v", err)
	}
}
