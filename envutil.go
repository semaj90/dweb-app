// envutil.go
// Centralized environment variable helpers to eliminate duplication.
// Keep tiny & stdlib-only so it can be imported by any build-tagged file safely.
package main

import (
	"os"
	"strconv"
	"time"
)

// Get string env with default.
func getEnv(key, def string) string {
    if v := os.Getenv(key); v != "" {
        return v
    }
    return def
}

// Get int env with default; non-parsable → default.
func getEnvInt(key string, def int) int {
    if v := os.Getenv(key); v != "" {
        if n, err := strconv.Atoi(v); err == nil {
            return n
        }
    }
    return def
}

// Get float env with default; non-parsable → default.
func getEnvFloat(key string, def float64) float64 {
    if v := os.Getenv(key); v != "" {
        if f, err := strconv.ParseFloat(v, 64); err == nil {
            return f
        }
    }
    return def
}

// Get bool env with default; accepts 1,true,yes,on (case-insensitive) for true.
func getEnvBool(key string, def bool) bool {
    if v := os.Getenv(key); v != "" {
        switch vLower := lower(v); vLower {
        case "1", "true", "yes", "on":
            return true
        case "0", "false", "no", "off":
            return false
        }
    }
    return def
}

// Get duration env with default; parses time.ParseDuration formats.
func getEnvDuration(key string, def time.Duration) time.Duration {
    if v := os.Getenv(key); v != "" {
        if d, err := time.ParseDuration(v); err == nil {
            return d
        }
    }
    return def
}

// minimal lower (avoid pulling in strings frequently in hot paths)
func lower(s string) string {
    b := []byte(s)
    for i, c := range b {
        if c >= 'A' && c <= 'Z' {
            b[i] = c + 32
        }
    }
    return string(b)
}
