package main

import (
	"crypto/tls"
	"fmt"
	"io"
	"net/http"
	"time"
)

func main() {
	fmt.Println("🧪 Testing QUIC Services Integration")
	fmt.Println("===================================")

	// Test each service port with regular HTTP/1.1 first
	testPorts := []struct{
		name string
		port int
		service string
	}{
		{"QUIC Legal Gateway", 8445, "Legal Gateway"},
		{"QUIC Vector Proxy", 8545, "Vector Proxy"}, 
		{"QUIC AI Stream", 8546, "AI Stream"},
	}

	for _, test := range testPorts {
		fmt.Printf("\n🔍 Testing %s on port %d\n", test.name, test.port)
		
		// Try with HTTPS (should work even if not HTTP/3)
		url := fmt.Sprintf("https://localhost:%d/health", test.port)
		
		client := &http.Client{
			Transport: &http.Transport{
				TLSClientConfig: &tls.Config{
					InsecureSkipVerify: true,
				},
			},
			Timeout: 3 * time.Second,
		}
		
		resp, err := client.Get(url)
		if err != nil {
			fmt.Printf("❌ HTTPS Error: %v\n", err)
			
			// Try HTTP (fallback)
			httpUrl := fmt.Sprintf("http://localhost:%d/health", test.port)
			resp, err = http.Get(httpUrl)
			if err != nil {
				fmt.Printf("❌ HTTP Error: %v\n", err)
				continue
			}
		}
		
		defer resp.Body.Close()
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			fmt.Printf("❌ Read error: %v\n", err)
			continue
		}

		fmt.Printf("✅ Status: %s\n", resp.Status)
		fmt.Printf("✅ Response: %s\n", string(body))
	}

	// Test integration with main system
	fmt.Println("\n🔗 Testing Integration with Main Legal AI System")
	fmt.Println("================================================")
	
	// Check if main services are accessible
	mainServices := []struct{
		name string
		url string
	}{
		{"Load Balancer", "http://localhost:8099/status"},
		{"Legal API", "http://localhost:8094/health"},
		{"AI API", "http://localhost:8095/health"},
		{"SvelteKit Frontend", "http://localhost:5173"},
	}

	for _, service := range mainServices {
		fmt.Printf("\n🔍 Testing %s\n", service.name)
		
		client := &http.Client{Timeout: 3 * time.Second}
		resp, err := client.Get(service.url)
		if err != nil {
			fmt.Printf("❌ Error: %v\n", err)
			continue
		}
		defer resp.Body.Close()
		
		fmt.Printf("✅ Status: %s\n", resp.Status)
		if resp.StatusCode == 200 {
			fmt.Printf("✅ Service: Healthy\n")
		}
	}

	fmt.Println("\n🎯 Integration Test Complete!")
}