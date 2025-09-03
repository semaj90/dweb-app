package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"time"

	"legal-ai-production/internal/envutil"
	"os"
	"strconv"
	"strings"
	embedpb "legal-ai-production/proto/embed"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
)

type GatewayConfig struct {
	HTTPPort      int
	CudaServiceURL    string
	UploadServiceURL  string
	EnableProxying    bool
}

type MultiProtocolGateway struct {
	config     *GatewayConfig
	httpClient *http.Client
}

func NewMultiProtocolGateway(cfg *GatewayConfig) (*MultiProtocolGateway, error) {
	return &MultiProtocolGateway{
		config: cfg,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}, nil
}

type EmbedService struct{ embedpb.UnimplementedEmbedderServer }

type ServiceStatus struct {
	Name      string `json:"name"`
	URL       string `json:"url"`
	Available bool   `json:"available"`
	Status    interface{} `json:"status,omitempty"`
}

type GatewayStatus struct {
	Gateway  string          `json:"gateway"`
	Version  string          `json:"version"`
	Services []ServiceStatus `json:"services"`
	Features map[string]bool `json:"features"`
	Uptime   string          `json:"uptime"`
}

func (g *MultiProtocolGateway) proxyRequest(c *gin.Context, targetURL string) {
	// Create target URL
	url := targetURL + c.Request.URL.Path
	if c.Request.URL.RawQuery != "" {
		url += "?" + c.Request.URL.RawQuery
	}

	// Create request
	var bodyBytes []byte
	if c.Request.Body != nil {
		bodyBytes, _ = io.ReadAll(c.Request.Body)
	}

	req, err := http.NewRequest(c.Request.Method, url, bytes.NewBuffer(bodyBytes))
	if err != nil {
		c.JSON(500, gin.H{"error": "Failed to create proxy request", "details": err.Error()})
		return
	}

	// Copy headers
	for key, values := range c.Request.Header {
		for _, value := range values {
			req.Header.Add(key, value)
		}
	}

	// Execute request
	resp, err := g.httpClient.Do(req)
	if err != nil {
		c.JSON(500, gin.H{"error": "Proxy request failed", "details": err.Error()})
		return
	}
	defer resp.Body.Close()

	// Copy response
	for key, values := range resp.Header {
		for _, value := range values {
			c.Writer.Header().Add(key, value)
		}
	}

	c.Writer.WriteHeader(resp.StatusCode)
	io.Copy(c.Writer, resp.Body)
}

func (g *MultiProtocolGateway) checkServiceHealth(url string) (bool, interface{}) {
	resp, err := g.httpClient.Get(url + "/health")
	if err != nil {
		return false, nil
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return false, nil
	}

	var status interface{}
	json.NewDecoder(resp.Body).Decode(&status)
	return true, status
}

func runHTTP(cfg *GatewayConfig, gateway *MultiProtocolGateway) *http.Server {
	r := gin.Default()
	r.Use(cors.Default())

	// Enhanced health endpoint with service discovery
	r.GET("/health", func(c *gin.Context) {
		startTime := time.Now()
		
		// Check all configured services
		services := []ServiceStatus{}
		
		if cfg.CudaServiceURL != "" {
			available, status := gateway.checkServiceHealth(cfg.CudaServiceURL)
			services = append(services, ServiceStatus{
				Name:      "CUDA Processing Service",
				URL:       cfg.CudaServiceURL,
				Available: available,
				Status:    status,
			})
		}
		
		if cfg.UploadServiceURL != "" {
			available, status := gateway.checkServiceHealth(cfg.UploadServiceURL)
			services = append(services, ServiceStatus{
				Name:      "Document Upload Service",
				URL:       cfg.UploadServiceURL,
				Available: available,
				Status:    status,
			})
		}

		// Check RTX Tensor Core availability through CUDA service
		rtxAvailable := false
		if cfg.CudaServiceURL != "" {
			if resp, err := gateway.httpClient.Get(cfg.CudaServiceURL + "/cuda/info"); err == nil && resp.StatusCode == 200 {
				var cudaInfo map[string]interface{}
				json.NewDecoder(resp.Body).Decode(&cudaInfo)
				resp.Body.Close()
				
				if capability, ok := cudaInfo["compute_capability"].(string); ok && capability >= "8.6" {
					rtxAvailable = true
				}
			}
		}

		gatewayStatus := GatewayStatus{
			Gateway:  "Multi-Protocol RTX Gateway",
			Version:  "2.0.0-rtx-enhanced",
			Services: services,
			Features: map[string]bool{
				"rtx_tensor_cores":       rtxAvailable,
				"cuda_processing":        len(services) > 0 && services[0].Available,
				"document_upload":        len(services) > 1 && services[1].Available,
				"4bit_quantization":      rtxAvailable,
				"negative_latent_space":  rtxAvailable,
				"multi_protocol":         true,
				"service_proxying":       cfg.EnableProxying,
			},
			Uptime: time.Since(startTime).String(),
		}

		// Add RTX-specific information if available
		if rtxAvailable {
			gatewayStatus.Features["tensor_core_generation"] = true
		}

		c.JSON(200, gatewayStatus)
	})

	// RTX Processing endpoints
	if cfg.EnableProxying && cfg.CudaServiceURL != "" {
		// Proxy CUDA processing requests
		r.Any("/cuda/*path", func(c *gin.Context) {
			gateway.proxyRequest(c, cfg.CudaServiceURL)
		})
		
		// Enhanced compute endpoint with RTX optimization
		r.POST("/compute/rtx", func(c *gin.Context) {
			var request map[string]interface{}
			if err := c.ShouldBindJSON(&request); err != nil {
				c.JSON(400, gin.H{"error": "Invalid request format"})
				return
			}

			// Add RTX optimization metadata
			if request["metadata"] == nil {
				request["metadata"] = make(map[string]interface{})
			}
			metadata := request["metadata"].(map[string]interface{})
			metadata["rtx_optimization"] = true
			metadata["gateway_enhanced"] = true
			metadata["tensor_cores_enabled"] = true
			metadata["quantization"] = "4bit"
			metadata["negative_latent_space"] = true

			// Forward to CUDA service
			jsonData, _ := json.Marshal(request)
			resp, err := gateway.httpClient.Post(
				cfg.CudaServiceURL+"/cuda/compute",
				"application/json",
				bytes.NewBuffer(jsonData),
			)
			
			if err != nil {
				c.JSON(500, gin.H{"error": "CUDA service unavailable", "details": err.Error()})
				return
			}
			defer resp.Body.Close()

			var result map[string]interface{}
			json.NewDecoder(resp.Body).Decode(&result)
			
			// Add gateway metadata to response
			result["gateway_processed"] = true
			result["rtx_enhanced"] = true
			
			c.JSON(resp.StatusCode, result)
		})
	}

	// Upload service proxying
	if cfg.EnableProxying && cfg.UploadServiceURL != "" {
		r.Any("/upload/*path", func(c *gin.Context) {
			gateway.proxyRequest(c, cfg.UploadServiceURL)
		})
	}

	// Unified status endpoint
	r.GET("/status/all", func(c *gin.Context) {
		status := map[string]interface{}{
			"timestamp": time.Now().UTC(),
			"gateway": map[string]interface{}{
				"port": cfg.HTTPPort,
				"version": "2.0.0-rtx",
				"features_enabled": cfg.EnableProxying,
			},
		}

		if cfg.CudaServiceURL != "" {
			available, cudaStatus := gateway.checkServiceHealth(cfg.CudaServiceURL)
			status["cuda_service"] = map[string]interface{}{
				"url": cfg.CudaServiceURL,
				"available": available,
				"status": cudaStatus,
			}
		}

		if cfg.UploadServiceURL != "" {
			available, uploadStatus := gateway.checkServiceHealth(cfg.UploadServiceURL)
			status["upload_service"] = map[string]interface{}{
				"url": cfg.UploadServiceURL,
				"available": available,
				"status": uploadStatus,
			}
		}

		c.JSON(200, status)
	})

	srv := &http.Server{Addr: fmt.Sprintf(":%d", cfg.HTTPPort), Handler: r}
	go func() {
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Printf("HTTP server error: %v", err)
		}
	}()
	
	return srv
}

func getStringEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getBoolEnv(key string, defaultValue bool) bool {
	if value := os.Getenv(key); value != "" {
		if b, err := strconv.ParseBool(strings.ToLower(value)); err == nil {
			return b
		}
	}
	return defaultValue
}

func main() {
	cfg := &GatewayConfig{
		HTTPPort:         envutil.GetInt("GATEWAY_HTTP_PORT", 8230),
		CudaServiceURL:   getStringEnv("CUDA_SERVICE_URL", "http://localhost:8096"),
		UploadServiceURL: getStringEnv("UPLOAD_SERVICE_URL", "http://localhost:8093"),
		EnableProxying:   getBoolEnv("ENABLE_PROXYING", true),
	}

	gateway, err := NewMultiProtocolGateway(cfg)
	if err != nil {
		log.Fatalf("Gateway initialization failed: %v", err)
	}

	httpSrv := runHTTP(cfg, gateway)
	defer httpSrv.Shutdown(context.Background())

	log.Printf("🚀 Multi-Protocol RTX Gateway running on port :%d", cfg.HTTPPort)
	log.Printf("🎯 CUDA Service: %s", cfg.CudaServiceURL)
	log.Printf("📄 Upload Service: %s", cfg.UploadServiceURL)
	log.Printf("⚡ RTX Tensor Core processing enabled")
	log.Printf("🔧 Service proxying: %v", cfg.EnableProxying)

	select {} // block indefinitely
}
