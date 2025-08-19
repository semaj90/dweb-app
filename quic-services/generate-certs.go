package main

import (
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"fmt"
	"math/big"
	"net"
	"os"
	"time"
)

func main() {
	// Create certs directory if it doesn't exist
	if err := os.MkdirAll("certs", 0755); err != nil {
		fmt.Printf("Failed to create certs directory: %v\n", err)
		return
	}

	// Generate private key
	priv, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		fmt.Printf("Failed to generate private key: %v\n", err)
		return
	}

	// Create certificate template
	template := x509.Certificate{
		SerialNumber: big.NewInt(1),
		Subject: pkix.Name{
			Organization:  []string{"Legal AI Development"},
			Country:       []string{"US"},
			Province:      []string{""},
			Locality:      []string{"Local"},
			StreetAddress: []string{""},
			PostalCode:    []string{""},
		},
		NotBefore:    time.Now(),
		NotAfter:     time.Now().Add(365 * 24 * time.Hour), // Valid for 1 year
		KeyUsage:     x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature,
		ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		IPAddresses:  []net.IP{net.IPv4(127, 0, 0, 1), net.IPv6loopback},
		DNSNames:     []string{"localhost"},
	}

	// Create certificate
	certDER, err := x509.CreateCertificate(rand.Reader, &template, &template, &priv.PublicKey, priv)
	if err != nil {
		fmt.Printf("Failed to create certificate: %v\n", err)
		return
	}

	// Save certificate
	certOut, err := os.Create("certs/server.crt")
	if err != nil {
		fmt.Printf("Failed to open cert.pem for writing: %v\n", err)
		return
	}
	defer certOut.Close()

	if err := pem.Encode(certOut, &pem.Block{Type: "CERTIFICATE", Bytes: certDER}); err != nil {
		fmt.Printf("Failed to write certificate: %v\n", err)
		return
	}

	// Save private key
	keyOut, err := os.Create("certs/server.key")
	if err != nil {
		fmt.Printf("Failed to open key.pem for writing: %v\n", err)
		return
	}
	defer keyOut.Close()

	privDER, err := x509.MarshalPKCS8PrivateKey(priv)
	if err != nil {
		fmt.Printf("Failed to marshal private key: %v\n", err)
		return
	}

	if err := pem.Encode(keyOut, &pem.Block{Type: "PRIVATE KEY", Bytes: privDER}); err != nil {
		fmt.Printf("Failed to write private key: %v\n", err)
		return
	}

	fmt.Println("✅ Development TLS certificates generated successfully")
	fmt.Println("📄 Certificate: certs/server.crt")
	fmt.Println("🔑 Private Key: certs/server.key")
}