package main

import (
	"crypto/rand"
	"crypto/rsa"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"fmt"
	"log"
	"math/big"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"strings"
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

func main() {
	// Configuration from environment
	listenAddr := ":" + env("QUIC_GATEWAY_PORT", "8443")
	backendURL := env("BACKEND_URL", "http://localhost:5173")
	enableHTTPFallback := strings.ToLower(env("ENABLE_HTTP_FALLBACK", "true")) == "true"
	httpFallbackAddr := ":" + env("HTTP_FALLBACK_PORT", "8444")

	// Parse backend URL
	target, err := url.Parse(backendURL)
	if err != nil {
		log.Fatalf("Invalid backend URL: %v", err)
	}

	// Create reverse proxy
	proxy := httputil.NewSingleHostReverseProxy(target)
	proxy.Director = func(req *http.Request) {
		req.URL.Scheme = target.Scheme
		req.URL.Host = target.Host
		req.Host = target.Host
		req.Header.Set("X-Forwarded-Proto", "h3")
		req.Header.Set("X-Forwarded-For", req.RemoteAddr)
	}

	// Create HTTP handler
	mux := http.NewServeMux()

	// Health check endpoint
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, `{"status":"ok","service":"quic-gateway","protocol":"http3"}`)
	})

	// Proxy all other requests
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		// Add QUIC-specific headers
		w.Header().Set("Alt-Svc", "h3=\":"+env("QUIC_GATEWAY_PORT", "8443")+"\"; ma=86400")
		proxy.ServeHTTP(w, r)
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
	log.Printf("🌐 QUIC Gateway listening on https://localhost%s (HTTP/3) -> %s", listenAddr, backendURL)
	if err := server.ListenAndServe(); err != nil {
		log.Fatalf("QUIC gateway error: %v", err)
	}
}