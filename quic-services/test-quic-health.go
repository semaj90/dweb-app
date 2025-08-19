package main

import (
	"crypto/tls"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/quic-go/quic-go/http3"
)

func main() {
	// Create HTTP/3 client with insecure TLS for development
	client := &http.Client{
		Transport: &http3.RoundTripper{
			TLSClientConfig: &tls.Config{
				InsecureSkipVerify: true,
			},
		},
		Timeout: 5 * time.Second,
	}

	// Test endpoints
	endpoints := []string{
		"https://localhost:8445/health", // QUIC Legal Gateway
		"https://localhost:8545/health", // QUIC Vector Proxy
		"https://localhost:8546/health", // QUIC AI Stream
	}

	fmt.Println("🧪 Testing QUIC Services Health Endpoints")
	fmt.Println("=========================================")

	for _, endpoint := range endpoints {
		fmt.Printf("\n🔍 Testing: %s\n", endpoint)
		
		resp, err := client.Get(endpoint)
		if err != nil {
			fmt.Printf("❌ Error: %v\n", err)
			continue
		}
		defer resp.Body.Close()

		body, err := io.ReadAll(resp.Body)
		if err != nil {
			fmt.Printf("❌ Read error: %v\n", err)
			continue
		}

		fmt.Printf("✅ Status: %s\n", resp.Status)
		fmt.Printf("✅ Protocol: %s\n", resp.Proto)
		fmt.Printf("✅ Response: %s\n", string(body))
	}

	fmt.Println("\n🎯 QUIC Health Check Complete!")
}