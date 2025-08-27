package main

import (
	"bytes"
	"crypto/rand"
	"crypto/rsa"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"io"
	"log"
	"math/big"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/quic-go/quic-go/http3"
)

func env(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

// loadDevCertificate generates a self-signed certificate for development
func loadDevCertificate() *tls.Config {
	// Generate RSA key
	key, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		log.Fatalf("Failed to generate RSA key: %v", err)
	}

	// Create certificate template
	template := &x509.Certificate{
		SerialNumber: big.NewInt(time.Now().UnixNano()),
		Subject: pkix.Name{
			Organization:  []string{"Legal AI Platform"},
			Country:       []string{"US"},
			Province:      []string{""},
			Locality:      []string{"Local"},
			StreetAddress: []string{""},
			PostalCode:    []string{""},
			CommonName:    "localhost",
		},
		NotBefore:             time.Now().Add(-time.Hour),
		NotAfter:              time.Now().Add(365 * 24 * time.Hour),
		KeyUsage:              x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		IPAddresses:           []net.IP{net.IPv4(127, 0, 0, 1), net.IPv6loopback},
		DNSNames:              []string{"localhost"},
		BasicConstraintsValid: true,
	}

	// Create the certificate
	certDER, err := x509.CreateCertificate(rand.Reader, template, template, &key.PublicKey, key)
	if err != nil {
		log.Fatalf("Failed to create certificate: %v", err)
	}

	// PEM encode
	certPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: certDER})
	keyPEM := pem.EncodeToMemory(&pem.Block{Type: "RSA PRIVATE KEY", Bytes: x509.MarshalPKCS1PrivateKey(key)})

	// Create TLS certificate
	cert, err := tls.X509KeyPair(certPEM, keyPEM)
	if err != nil {
		log.Fatalf("Failed to create TLS certificate: %v", err)
	}

	return &tls.Config{
		Certificates: []tls.Certificate{cert},
		NextProtos:   []string{"h3", "http/1.1"},
		ServerName:   "localhost",
	}
}

// VectorCache provides intelligent caching for vector operations
type VectorCache struct {
	mu    sync.RWMutex
	items map[string]VectorCacheEntry
	ttl   time.Duration
}

type VectorCacheEntry struct {
	data      []byte
	expires   time.Time
	hitCount  int
	createdAt time.Time
}

func NewVectorCache() *VectorCache {
	ttlStr := env("VECTOR_CACHE_TTL", "5m")
	ttl, err := time.ParseDuration(ttlStr)
	if err != nil {
		ttl = 5 * time.Minute
	}

	return &VectorCache{
		items: make(map[string]VectorCacheEntry),
		ttl:   ttl,
	}
}

func (vc *VectorCache) Get(key string) ([]byte, bool) {
	vc.mu.RLock()
	entry, exists := vc.items[key]
	vc.mu.RUnlock()

	if !exists || time.Now().After(entry.expires) {
		if exists {
			vc.mu.Lock()
			delete(vc.items, key)
			vc.mu.Unlock()
		}
		return nil, false
	}

	// Update hit count
	vc.mu.Lock()
	entry.hitCount++
	vc.items[key] = entry
	vc.mu.Unlock()

	return entry.data, true
}

func (vc *VectorCache) Set(key string, data []byte) {
	vc.mu.Lock()
	vc.items[key] = VectorCacheEntry{
		data:      data,
		expires:   time.Now().Add(vc.ttl),
		hitCount:  1,
		createdAt: time.Now(),
	}
	vc.mu.Unlock()
}

type VectorRequest struct {
	Query       string                 `json:"query"`
	Model       string                 `json:"model"`
	Limit       int                    `json:"limit"`
	Threshold   float64                `json:"threshold"`
	Metadata    map[string]interface{} `json:"metadata"`
	CacheKey    string                 `json:"cache_key"`
	UseCache    bool                   `json:"use_cache"`
	Operation   string                 `json:"operation"`
	Dimensions  int                    `json:"dimensions"`
}

type VectorResponse struct {
	Results     []VectorResult `json:"results"`
	Took        string         `json:"took"`
	Total       int            `json:"total"`
	CacheHit    bool           `json:"cache_hit"`
	Model       string         `json:"model"`
	Operation   string         `json:"operation"`
}

type VectorResult struct {
	ID       string                 `json:"id"`
	Score    float64                `json:"score"`
	Content  string                 `json:"content"`
	Metadata map[string]interface{} `json:"metadata"`
}

func main() {
	// Configuration
	listenAddr := ":" + env("QUIC_VECTOR_PORT", "8445")
	qdrantURL := env("QDRANT_URL", "http://localhost:6333")
	pgvectorURL := env("PGVECTOR_URL", "postgresql://postgres:postgres@localhost:5432/legal_ai_db")
	enableHTTPFallback := strings.ToLower(env("ENABLE_HTTP_FALLBACK", "true")) == "true"
	httpFallbackAddr := ":" + env("HTTP_FALLBACK_PORT", "8446")

	// Initialize vector cache
	vectorCache := NewVectorCache()

	// Parse backend URLs
	qdrantTarget, err := url.Parse(qdrantURL)
	if err != nil {
		log.Fatalf("Invalid Qdrant URL: %v", err)
	}

	// Create reverse proxy for Qdrant
	qdrantProxy := httputil.NewSingleHostReverseProxy(qdrantTarget)
	qdrantProxy.Director = func(req *http.Request) {
		req.URL.Scheme = qdrantTarget.Scheme
		req.URL.Host = qdrantTarget.Host
		req.Host = qdrantTarget.Host
		req.Header.Set("X-Forwarded-Proto", "h3")
		req.Header.Set("X-Forwarded-For", req.RemoteAddr)
	}

	// Create HTTP handler
	mux := http.NewServeMux()

	// Health check endpoint
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, `{"status":"ok","service":"quic-vector-proxy","protocol":"http3","backends":{"qdrant":"%s","pgvector":"%s"}}`, qdrantURL, pgvectorURL)
	})

	// Vector search endpoint with intelligent caching
	mux.HandleFunc("/api/vector/search", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		start := time.Now()

		var req VectorRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid JSON", http.StatusBadRequest)
			return
		}

		// Generate cache key if not provided
		if req.CacheKey == "" {
			req.CacheKey = fmt.Sprintf("vector:%s:%s:%d:%f", req.Query, req.Model, req.Limit, req.Threshold)
		}

		// Check cache first
		if req.UseCache {
			if cached, found := vectorCache.Get(req.CacheKey); found {
				w.Header().Set("Content-Type", "application/json")
				w.Header().Set("X-Cache", "HIT")
				w.Header().Set("X-Cache-Key", req.CacheKey)
				w.Write(cached)
				return
			}
		}

		// Forward to appropriate backend (Qdrant or pgvector)
		var backend string
		var proxyReq *http.Request

		if strings.Contains(strings.ToLower(req.Operation), "qdrant") {
			backend = "qdrant"
			proxyReq, _ = http.NewRequest("POST", qdrantURL+"/collections/legal_documents/points/search", bytes.NewBuffer([]byte{}))
		} else {
			backend = "pgvector"
			// For pgvector, we'll proxy to the enhanced RAG service
			enhancedRAGURL := env("ENHANCED_RAG_URL", "http://localhost:8094")
			proxyReq, _ = http.NewRequest("POST", enhancedRAGURL+"/api/vector/search", bytes.NewBuffer([]byte{}))
		}

		// Copy headers
		for name, values := range r.Header {
			for _, value := range values {
				proxyReq.Header.Add(name, value)
			}
		}

		// Execute the request
		client := &http.Client{Timeout: 30 * time.Second}
		resp, err := client.Do(proxyReq)
		if err != nil {
			http.Error(w, "Backend error", http.StatusInternalServerError)
			return
		}
		defer resp.Body.Close()

		// Read response
		respData, err := io.ReadAll(resp.Body)
		if err != nil {
			http.Error(w, "Failed to read response", http.StatusInternalServerError)
			return
		}

		// Enhance response with metadata
		var vectorResp VectorResponse
		json.Unmarshal(respData, &vectorResp)
		vectorResp.Took = time.Since(start).String()
		vectorResp.CacheHit = false
		vectorResp.Model = req.Model
		vectorResp.Operation = req.Operation

		// Re-encode response
		enhancedResp, _ := json.Marshal(vectorResp)

		// Cache the result if requested
		if req.UseCache && resp.StatusCode == 200 {
			vectorCache.Set(req.CacheKey, enhancedResp)
		}

		// Return response
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-Cache", "MISS")
		w.Header().Set("X-Backend", backend)
		w.Header().Set("X-Cache-Key", req.CacheKey)
		w.WriteHeader(resp.StatusCode)
		w.Write(enhancedResp)
	})

	// Proxy Qdrant requests
	mux.HandleFunc("/qdrant/", func(w http.ResponseWriter, r *http.Request) {
		// Strip /qdrant prefix
		r.URL.Path = strings.TrimPrefix(r.URL.Path, "/qdrant")
		w.Header().Set("Alt-Svc", "h3=\":"+env("QUIC_VECTOR_PORT", "8445")+"\"; ma=86400")
		qdrantProxy.ServeHTTP(w, r)
	})

	// Default handler
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "QUIC Vector Proxy - Use /api/vector/search or /qdrant/ endpoints", http.StatusNotFound)
	})

	// Generate TLS config
	tlsConfig := loadDevCertificate()

	// Start HTTP/3 server
	server := &http3.Server{
		Handler:   mux,
		Addr:      listenAddr,
		TLSConfig: tlsConfig,
	}

	// Optional HTTP/2 fallback
	if enableHTTPFallback {
		go func() {
			log.Printf("🔁 HTTP/2 fallback listening on http://localhost%s", httpFallbackAddr)
			fallbackServer := &http.Server{
				Addr:      httpFallbackAddr,
				Handler:   mux,
				TLSConfig: tlsConfig,
			}
			if err := fallbackServer.ListenAndServeTLS("", ""); err != nil && err != http.ErrServerClosed {
				log.Printf("HTTP/2 fallback error: %v", err)
			}
		}()
	}

	// Start QUIC server
	log.Printf("🧠 QUIC Vector Proxy listening on https://localhost%s (HTTP/3)", listenAddr)
	log.Printf("   Backends: Qdrant (%s), pgvector (%s)", qdrantURL, pgvectorURL)
	if err := server.ListenAndServe(); err != nil {
		log.Fatalf("QUIC vector proxy error: %v", err)
	}
}