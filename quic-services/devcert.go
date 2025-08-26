package main

// Centralized development self-signed certificate (PEM) used by QUIC services.
// Avoids duplicate generateSelfSignedCert() definitions causing redeclaration errors.
// In production, replace with real certificates loaded from secure storage / env.

import (
	"crypto/tls"
	"crypto/x509"
	"encoding/pem"
	"errors"
)

// NOTE: These PEM blocks are placeholders (truncated). Replace with proper
// development cert/key material or dynamically generate if needed. They only
// need to parse for local testing; NOT for production use.
var (
    devCertPEM = []byte(`-----BEGIN CERTIFICATE-----\nMIIBwjCCAWigAwIBAgIRANJTESTDEVONLYCERTMB4XDTI1MDEwMTAwMDAwMFoXDTMwMDEwMTAwMDAwMFowEjEQMA4GA1UEAwwHZGV2LWxvYzCBnzANBgkqhkiG9w0BAQEFAAOBjQAwgYkCgYEAp3QrT+0GpPj6JtI6GJxfxYXYbzG4zCN6bS1QPmXw7mSE0tBAvzkdxl3p7ZozKx6CxInpnPclmFasqYIpWHqxkB2WfJuyQG3R2FxlrF9SoqApqI4B6JeBtFtWz3FQAmQiAlAA6PS+PfDOJv6JfsDqYZI6zlkzsDycQAriS/4IYVAvtcCAwEAAaNTMFEwHQYDVR0OBBYEFIU3BjsU9o06bUi0n8O9q1vOwYkrMB8GA1UdIwQYMBaAFIU3BjsU9o06bUi0n8O9q1vOwYkrMA8GA1UdEwEB/wQFMAMBAf8wDQYJKoZIhvcNAQELBQADgYEAKWyD6lyWtrbOd6dvC9O9+P1VDTQU7zmtGI44UPZflBqP2b4c0tXynA2bE0cLPp9+ZxKrtEKqcN3JJqualDEVONLYTRUNCATED\n-----END CERTIFICATE-----`)
    devKeyPEM = []byte(`-----BEGIN RSA PRIVATE KEY-----\nMIIBOgIBAAJBAKd0K0/tBqT4+i bSOhicX8WF2G8xuMwjem0tUD5l8O5khNLQQL85HcZd6e2aMysegsSJ6Zz3JZhWrKmCKVh6sCAwEAAQJAK9SGkR3T1vl9lwvR2vl9Ju n2T2yupj85bQzjwr8erG1NdD4C+3zzDEVONLYTRUNCATED\n-----END RSA PRIVATE KEY-----`)
)

// loadDevCertificate returns a tls.Certificate built from the embedded
// development certificate/key. Panics on error to surface issues early.
func loadDevCertificate() tls.Certificate {
    cert, err := tls.X509KeyPair(devCertPEM, devKeyPEM)
    if err != nil {
        panic(err)
    }
    return cert
}

// parseCertForDebug extracts the leaf certificate for optional logging.
func parseCertForDebug(cert tls.Certificate) (*x509.Certificate, error) {
    if len(cert.Certificate) == 0 {
        return nil, errors.New("no certificate data")
    }
    return x509.ParseCertificate(cert.Certificate[0])
}

// exportDevCertPEM returns the certificate PEM for tooling endpoints.
func exportDevCertPEM() []byte { return pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: mustFirstCert(loadDevCertificate())}) }

func mustFirstCert(c tls.Certificate) []byte { return c.Certificate[0] }
