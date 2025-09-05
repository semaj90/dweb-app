package main

import (
	"bytes"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"io"
	"log"
	"math/big"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/quic-go/quic-go/http3"
)

func env(k, d string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return d
}

// Simple in-memory edge cache
type cacheEntry struct {
	status  int
	header  http.Header
	body    []byte
	expires time.Time
}

type edgeCache struct {
	mu       sync.RWMutex
	items    map[string]cacheEntry
	ttl      time.Duration
	maxItems int
	maxBytes int
}

func newEdgeCacheFromEnv() *edgeCache {
	ttl, err := time.ParseDuration(env("QUIC_EDGE_CACHE_TTL", "30s"))
	if err != nil {
		ttl = 30 * time.Second
	}
	maxItems := 500
	if s := os.Getenv("QUIC_EDGE_CACHE_MAX"); s != "" {
		if n, err := strconv.Atoi(s); err == nil {
			maxItems = n
		}
	}
	maxBytes := 1_048_576
	if s := os.Getenv("QUIC_EDGE_CACHE_MAX_BYTES"); s != "" {
		if n, err := strconv.Atoi(s); err == nil {
			maxBytes = n
		}
	}
	return &edgeCache{items: make(map[string]cacheEntry), ttl: ttl, maxItems: maxItems, maxBytes: maxBytes}
}

func (c *edgeCache) get(k string) (cacheEntry, bool) {
	c.mu.RLock()
	e, ok := c.items[k]
	c.mu.RUnlock()
	if !ok {
		return cacheEntry{}, false
	}
	if time.Now().After(e.expires) {
		c.mu.Lock()
		delete(c.items, k)
		c.mu.Unlock()
		return cacheEntry{}, false
	}
	return e, true
}

// peek returns entry and whether it exists and whether it's expired
func (c *edgeCache) peek(k string) (cacheEntry, bool, bool) {
	c.mu.RLock()
	e, ok := c.items[k]
	c.mu.RUnlock()
	if !ok {
		return cacheEntry{}, false, false
	}
	return e, true, time.Now().After(e.expires)
}
func (c *edgeCache) set(k string, e cacheEntry) {
	if len(e.body) > c.maxBytes {
		return
	}
	c.mu.Lock()
	if len(c.items) >= c.maxItems {
		for kk := range c.items {
			delete(c.items, kk)
			break
		}
	}
	e.expires = time.Now().Add(c.ttl)
	c.items[k] = e
	c.mu.Unlock()
}
func (c *edgeCache) refresh(k string) {
	c.mu.Lock()
	if e, ok := c.items[k]; ok {
		e.expires = time.Now().Add(c.ttl)
		c.items[k] = e
	}
	c.mu.Unlock()
}

func shortHex(b []byte) string {
	const hx = "0123456789abcdef"
	out := make([]byte, len(b)*2)
	for i, v := range b {
		out[i*2] = hx[v>>4]
		out[i*2+1] = hx[v&0x0f]
	}
	return string(out)
}

func cacheKey(r *http.Request, body []byte) string {
	sum := sha256.Sum256(append([]byte(r.Method+"|"+r.URL.Path+"?"+r.URL.RawQuery+"|"), body...))
	return "edge:" + r.Host + ":" + shortHex(sum[:8])
}

func shouldCache(r *http.Request) bool {
	if r.Method != http.MethodGet && r.Method != http.MethodPost {
		return false
	}
	if strings.Contains(strings.ToLower(r.Header.Get("Accept")), "text/event-stream") {
		return false
	}
	if strings.ToLower(r.Header.Get("Upgrade")) != "" {
		return false
	}
	return true
}

type capWriter struct {
	http.ResponseWriter
	status int
	buf    bytes.Buffer
}

func (w *capWriter) WriteHeader(code int) { w.status = code; w.ResponseWriter.WriteHeader(code) }
func (w *capWriter) Write(p []byte) (int, error) {
	if w.buf.Len() < 2_000_000 {
		w.buf.Write(p)
	}
	return w.ResponseWriter.Write(p)
}

// generateSelfSignedTLS returns a TLS config with an in-memory self-signed certificate.
func generateSelfSignedTLS() *tls.Config {
	// Generate RSA key
	key, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		log.Fatalf("tls key: %v", err)
	}

	// Create certificate template
	tmpl := &x509.Certificate{
		SerialNumber:          big.NewInt(time.Now().UnixNano()),
		Subject:               pkix.Name{CommonName: "rag-quic-proxy"},
		NotBefore:             time.Now().Add(-time.Hour),
		NotAfter:              time.Now().Add(365 * 24 * time.Hour),
		KeyUsage:              x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		BasicConstraintsValid: true,
	}

	certDER, err := x509.CreateCertificate(rand.Reader, tmpl, tmpl, &key.PublicKey, key)
	if err != nil {
		log.Fatalf("tls cert: %v", err)
	}

	// PEM encode cert and key
	certPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: certDER})
	keyPEM := pem.EncodeToMemory(&pem.Block{Type: "RSA PRIVATE KEY", Bytes: x509.MarshalPKCS1PrivateKey(key)})
	cert, err := tls.X509KeyPair(certPEM, keyPEM)
	if err != nil {
		log.Fatalf("tls pair: %v", err)
	}

	return &tls.Config{Certificates: []tls.Certificate{cert}, NextProtos: []string{http3.NextProtoH3}}
}

func main() {
	backendURL := env("RAG_BACKEND_URL", "http://localhost:8093")
	frontAddr := ":" + env("RAG_QUIC_FRONT_PORT", "8443")
	fallbackAddr := ":" + env("RAG_QUIC_FALLBACK_PORT", "8444")
	enableFallback := strings.ToLower(env("RAG_QUIC_ENABLE_FALLBACK", "true")) == "true"

	target, err := url.Parse(backendURL)
	if err != nil {
		log.Fatalf("invalid backend url: %v", err)
	}

	rp := httputil.NewSingleHostReverseProxy(target)
	// Set headers helpful for HTTP/3 and tracing
	rp.Director = func(req *http.Request) {
		req.URL.Scheme = target.Scheme
		req.URL.Host = target.Host
		// Preserve original path and query
		req.Host = target.Host
		req.Header.Set("X-Forwarded-Proto", "h3")
		req.Header.Set("X-Forwarded-For", req.RemoteAddr)
		req.Header.Set("Alt-Svc", `h3=\":`+env("RAG_QUIC_FRONT_PORT", "8443")+`\"; ma=86400`)
	}

	mux := http.NewServeMux()
	cache := newEdgeCacheFromEnv()
	// metrics
	edgeHits := prometheus.NewCounter(prometheus.CounterOpts{Name: "edge_cache_hits_total", Help: "Total edge cache hits"})
	edgeMiss := prometheus.NewCounter(prometheus.CounterOpts{Name: "edge_cache_misses_total", Help: "Total edge cache misses"})
	edgeBytes := prometheus.NewCounter(prometheus.CounterOpts{Name: "edge_bytes_served_total", Help: "Total bytes served by proxy"})
	prometheus.MustRegister(edgeHits, edgeMiss, edgeBytes)
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"status":"ok","via":"quic-proxy"}`))
	})
	// metrics endpoint
	mux.Handle("/metrics", promhttp.Handler())

	// Proxy all other routes to backend with simple JSON edge cache and ETag revalidation for GET
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if !shouldCache(r) {
			rp.ServeHTTP(w, r)
			return
		}
		var bodyCopy []byte
		if r.Method == http.MethodPost && r.Body != nil {
			b, _ := io.ReadAll(r.Body)
			r.Body.Close()
			r.Body = io.NopCloser(bytes.NewReader(b))
			bodyCopy = b
		}
		key := cacheKey(r, bodyCopy)
		if e, ok := cache.get(key); ok {
			for k, vals := range e.header {
				for _, v := range vals {
					w.Header().Add(k, v)
				}
			}
			w.Header().Set("X-Cache", "H3-Edge-HIT")
			w.WriteHeader(e.status)
			_, _ = w.Write(e.body)
			edgeHits.Inc()
			edgeBytes.Add(float64(len(e.body)))
			return
		}
		// Conditional GET revalidation if we have a stale entry
		if r.Method == http.MethodGet {
			if stale, ok, expired := cache.peek(key); ok && expired {
				etag := stale.header.Get("ETag")
				if etag != "" {
					// Build backend URL
					backendURL := target.Scheme + "://" + target.Host + r.URL.RequestURI()
					req, _ := http.NewRequest(http.MethodGet, backendURL, nil)
					req.Header = r.Header.Clone()
					req.Header.Set("If-None-Match", etag)
					client := &http.Client{Timeout: 10 * time.Second}
					if resp, err := client.Do(req); err == nil {
						defer resp.Body.Close()
						if resp.StatusCode == http.StatusNotModified {
							for k, vals := range stale.header {
								for _, v := range vals {
									w.Header().Add(k, v)
								}
							}
							w.Header().Set("X-Cache", "H3-Edge-REVALIDATED")
							w.WriteHeader(http.StatusOK)
							_, _ = w.Write(stale.body)
							cache.refresh(key)
							edgeHits.Inc()
							edgeBytes.Add(float64(len(stale.body)))
							return
						}
						// Replace with fresh content if JSON 200
						buf, _ := io.ReadAll(resp.Body)
						for k := range w.Header() {
							w.Header().Del(k)
						}
						for k, vals := range resp.Header {
							for _, v := range vals {
								w.Header().Add(k, v)
							}
						}
						w.WriteHeader(resp.StatusCode)
						_, _ = w.Write(buf)
						edgeBytes.Add(float64(len(buf)))
						ct := strings.ToLower(resp.Header.Get("Content-Type"))
						if resp.StatusCode == 200 && strings.Contains(ct, "application/json") {
							hdrCopy := http.Header{}
							for k, vals := range resp.Header {
								vv := make([]string, len(vals))
								copy(vv, vals)
								hdrCopy[k] = vv
							}
							cache.set(key, cacheEntry{status: 200, header: hdrCopy, body: buf})
						}
						return
					}
				}
			}
		}
		cw := &capWriter{ResponseWriter: w, status: http.StatusOK}
		rp.ServeHTTP(cw, r)
		edgeMiss.Inc()
		edgeBytes.Add(float64(cw.buf.Len()))
		ct := strings.ToLower(w.Header().Get("Content-Type"))
		if cw.status == 200 && strings.Contains(ct, "application/json") {
			hdrCopy := http.Header{}
			for k, vals := range w.Header() {
				vv := make([]string, len(vals))
				copy(vv, vals)
				hdrCopy[k] = vv
			}
			cache.set(key, cacheEntry{status: cw.status, header: hdrCopy, body: cw.buf.Bytes()})
		}
	})

	tlsCfg := generateSelfSignedTLS()

	// Optional HTTPS fallback for clients without HTTP/3 support
	if enableFallback {
		httpsSrv := &http.Server{Addr: fallbackAddr, Handler: mux, TLSConfig: tlsCfg}
		go func() {
			log.Printf("🔁 HTTPS fallback listening on https://localhost%v (HTTP/1.1) -> %s", fallbackAddr, backendURL)
			if err := httpsSrv.ListenAndServeTLS("", ""); err != nil && err != http.ErrServerClosed {
				log.Printf("https fallback error: %v", err)
			}
		}()
	}

	srv := &http3.Server{Handler: mux, Addr: frontAddr, TLSConfig: tlsCfg}
	log.Printf("🔐 QUIC proxy listening on https://localhost%v (HTTP/3) -> %s", frontAddr, backendURL)
	if err := srv.ListenAndServe(); err != nil {
		log.Fatalf("quic proxy error: %v", err)
	}
}
