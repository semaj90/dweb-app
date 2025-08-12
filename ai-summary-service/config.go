// Configuration for Live Agent Enhanced Service
package main

import "time"

// Configuration structure
type Config struct {
	Port            string
	RedisAddr       string
	RedisPassword   string
	RedisDB         int
	PostgresURL     string
	OllamaURL       string
	EnableGPU       bool
	MaxConcurrency  int
	CacheExpiration time.Duration
	ModelContext    int
	Temperature     float64
}

// Default configuration
func DefaultConfig() *Config {
	return &Config{
		Port:            "8081",
		OllamaURL:       "http://localhost:11434",
		EnableGPU:       true,
		MaxConcurrency:  16,
		CacheExpiration: 5 * time.Minute,
		ModelContext:    4096,
		Temperature:     0.7,
	}
}