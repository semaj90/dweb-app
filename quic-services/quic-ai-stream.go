package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"crypto/rsa"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"log"
	"math/big"
	"net"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"
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

// StreamManager manages AI streaming sessions
type StreamManager struct {
	mu        sync.RWMutex
	sessions  map[string]*StreamSession
	upgrader  websocket.Upgrader
	ollamaURL string
	ragURL    string
}

type StreamSession struct {
	ID          string                 `json:"id"`
	CreatedAt   time.Time              `json:"created_at"`
	LastActive  time.Time              `json:"last_active"`
	Model       string                 `json:"model"`
	Context     []string               `json:"context"`
	Metadata    map[string]interface{} `json:"metadata"`
	conn        *websocket.Conn
	cancel      context.CancelFunc
}

type StreamRequest struct {
	Model     string                 `json:"model"`
	Prompt    string                 `json:"prompt"`
	Stream    bool                   `json:"stream"`
	Context   []string               `json:"context"`
	Options   map[string]interface{} `json:"options"`
	SessionID string                 `json:"session_id"`
	UseRAG    bool                   `json:"use_rag"`
	Metadata  map[string]interface{} `json:"metadata"`
}

type StreamChunk struct {
	Type      string                 `json:"type"`
	Content   string                 `json:"content"`
	Done      bool                   `json:"done"`
	SessionID string                 `json:"session_id"`
	Model     string                 `json:"model"`
	Metadata  map[string]interface{} `json:"metadata"`
	Timestamp time.Time              `json:"timestamp"`
}

func NewStreamManager() *StreamManager {
	return &StreamManager{
		sessions:  make(map[string]*StreamSession),
		ollamaURL: env("OLLAMA_URL", "http://localhost:11434"),
		ragURL:    env("ENHANCED_RAG_URL", "http://localhost:8094"),
		upgrader: websocket.Upgrader{
			CheckOrigin: func(r *http.Request) bool {
				return true // Allow all origins in development
			},
		},
	}
}

func (sm *StreamManager) CreateSession(model string) *StreamSession {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	sessionID := fmt.Sprintf("stream_%d", time.Now().UnixNano())
	_, cancel := context.WithCancel(context.Background())

	session := &StreamSession{
		ID:         sessionID,
		CreatedAt:  time.Now(),
		LastActive: time.Now(),
		Model:      model,
		Context:    make([]string, 0),
		Metadata:   make(map[string]interface{}),
		cancel:     cancel,
	}

	sm.sessions[sessionID] = session
	return session
}

func (sm *StreamManager) GetSession(sessionID string) (*StreamSession, bool) {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	session, exists := sm.sessions[sessionID]
	if exists {
		session.LastActive = time.Now()
	}
	return session, exists
}

func (sm *StreamManager) HandleWebSocket(w http.ResponseWriter, r *http.Request) {
	conn, err := sm.upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("WebSocket upgrade error: %v", err)
		return
	}
	defer conn.Close()

	// Create new session
	model := r.URL.Query().Get("model")
	if model == "" {
		model = "gemma3-legal"
	}

	session := sm.CreateSession(model)
	session.conn = conn

	log.Printf("WebSocket session created: %s (model: %s)", session.ID, model)

	// Handle messages
	for {
		var req StreamRequest
		err := conn.ReadJSON(&req)
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				log.Printf("WebSocket error: %v", err)
			}
			break
		}

		// Process streaming request
		go sm.processStreamRequest(session, &req)
	}

	// Cleanup
	sm.mu.Lock()
	delete(sm.sessions, session.ID)
	sm.mu.Unlock()
	session.cancel()
}

func (sm *StreamManager) processStreamRequest(session *StreamSession, req *StreamRequest) {
	// Send initial response
	chunk := &StreamChunk{
		Type:      "start",
		Content:   "",
		Done:      false,
		SessionID: session.ID,
		Model:     req.Model,
		Timestamp: time.Now(),
		Metadata:  map[string]interface{}{"status": "processing"},
	}
	session.conn.WriteJSON(chunk)

	// Choose backend based on request
	var backendURL string
	var payload []byte

	if req.UseRAG {
		// Enhanced RAG service
		backendURL = sm.ragURL + "/api/rag/stream"
		ragPayload := map[string]interface{}{
			"query":   req.Prompt,
			"model":   req.Model,
			"stream":  true,
			"context": req.Context,
			"options": req.Options,
		}
		payload, _ = json.Marshal(ragPayload)
	} else {
		// Direct Ollama streaming
		backendURL = sm.ollamaURL + "/api/generate"
		ollamaPayload := map[string]interface{}{
			"model":  req.Model,
			"prompt": req.Prompt,
			"stream": true,
			"context": req.Context,
			"options": req.Options,
		}
		payload, _ = json.Marshal(ollamaPayload)
	}

	// Make streaming request to backend
	httpReq, err := http.NewRequest("POST", backendURL, bytes.NewBuffer(payload))
	if err != nil {
		sm.sendError(session, fmt.Sprintf("Failed to create request: %v", err))
		return
	}

	httpReq.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 5 * time.Minute}
	resp, err := client.Do(httpReq)
	if err != nil {
		sm.sendError(session, fmt.Sprintf("Backend request failed: %v", err))
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		sm.sendError(session, fmt.Sprintf("Backend error: %d", resp.StatusCode))
		return
	}

	// Stream response back to client
	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}

		// Parse response chunk
		var backendChunk map[string]interface{}
		if err := json.Unmarshal([]byte(line), &backendChunk); err != nil {
			continue
		}

		// Convert to our chunk format
		chunk := &StreamChunk{
			Type:      "chunk",
			SessionID: session.ID,
			Model:     req.Model,
			Timestamp: time.Now(),
			Metadata:  req.Metadata,
		}

		if content, ok := backendChunk["response"].(string); ok {
			chunk.Content = content
		}

		if done, ok := backendChunk["done"].(bool); ok {
			chunk.Done = done
		}

		// Send chunk to client
		if err := session.conn.WriteJSON(chunk); err != nil {
			log.Printf("Failed to send chunk: %v", err)
			return
		}

		if chunk.Done {
			break
		}
	}

	// Send final chunk
	finalChunk := &StreamChunk{
		Type:      "done",
		Content:   "",
		Done:      true,
		SessionID: session.ID,
		Model:     req.Model,
		Timestamp: time.Now(),
		Metadata:  map[string]interface{}{"status": "completed"},
	}
	session.conn.WriteJSON(finalChunk)
}

func (sm *StreamManager) sendError(session *StreamSession, message string) {
	errorChunk := &StreamChunk{
		Type:      "error",
		Content:   message,
		Done:      true,
		SessionID: session.ID,
		Timestamp: time.Now(),
		Metadata:  map[string]interface{}{"error": true},
	}
	session.conn.WriteJSON(errorChunk)
}

func main() {
	// Configuration
	listenAddr := ":" + env("QUIC_AI_STREAM_PORT", "8447")
	enableHTTPFallback := strings.ToLower(env("ENABLE_HTTP_FALLBACK", "true")) == "true"
	httpFallbackAddr := ":" + env("HTTP_FALLBACK_PORT", "8448")

	// Initialize stream manager
	streamManager := NewStreamManager()

	// Create HTTP handler
	mux := http.NewServeMux()

	// Health check endpoint
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, `{"status":"ok","service":"quic-ai-stream","protocol":"http3","backends":{"ollama":"%s","rag":"%s"}}`, streamManager.ollamaURL, streamManager.ragURL)
	})

	// WebSocket streaming endpoint
	mux.HandleFunc("/ws/stream", streamManager.HandleWebSocket)

	// HTTP streaming endpoint (fallback)
	mux.HandleFunc("/api/stream", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req StreamRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid JSON", http.StatusBadRequest)
			return
		}

		// Set headers for SSE
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.Header().Set("Access-Control-Allow-Origin", "*")

		// Create session
		session := streamManager.CreateSession(req.Model)
		defer func() {
			streamManager.mu.Lock()
			delete(streamManager.sessions, session.ID)
			streamManager.mu.Unlock()
			session.cancel()
		}()

		// Process in goroutine and stream via SSE
		// For simplicity, this is a basic implementation
		// Full implementation would handle streaming properly
		fmt.Fprintf(w, "data: %s\n\n", `{"type":"start","session_id":"`+session.ID+`"}`)
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}

		fmt.Fprintf(w, "data: %s\n\n", `{"type":"done","session_id":"`+session.ID+`"}`)
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
	})

	// Session management endpoints
	mux.HandleFunc("/api/sessions", func(w http.ResponseWriter, r *http.Request) {
		streamManager.mu.RLock()
		sessions := make([]*StreamSession, 0, len(streamManager.sessions))
		for _, session := range streamManager.sessions {
			sessionCopy := *session
			sessionCopy.conn = nil // Don't serialize connection
			sessions = append(sessions, &sessionCopy)
		}
		streamManager.mu.RUnlock()

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(sessions)
	})

	// Default handler
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "QUIC AI Stream - Use /ws/stream for WebSocket or /api/stream for HTTP streaming", http.StatusNotFound)
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
	log.Printf("🤖 QUIC AI Stream listening on https://localhost%s (HTTP/3)", listenAddr)
	log.Printf("   WebSocket: wss://localhost%s/ws/stream", listenAddr)
	log.Printf("   HTTP Stream: https://localhost%s/api/stream", listenAddr)
	log.Printf("   Backends: Ollama (%s), RAG (%s)", streamManager.ollamaURL, streamManager.ragURL)
	if err := server.ListenAndServe(); err != nil {
		log.Fatalf("QUIC AI stream error: %v", err)
	}
}