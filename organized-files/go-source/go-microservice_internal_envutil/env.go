// Package envutil provides centralized environment configuration utilities
package envutil

import (
	"os"
	"strconv"
)

// Config represents environment configuration
type Config struct{}

// LoadConfig creates a new configuration instance
func LoadConfig() *Config {
	return &Config{}
}

// GetString returns environment variable value or default
func (c *Config) GetString(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

// GetInt returns environment variable as integer or default
func (c *Config) GetInt(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if intVal, err := strconv.Atoi(value); err == nil {
			return intVal
		}
	}
	return defaultValue
}

// GetBool returns environment variable as boolean or default
func (c *Config) GetBool(key string, defaultValue bool) bool {
	if value := os.Getenv(key); value != "" {
		if boolVal, err := strconv.ParseBool(value); err == nil {
			return boolVal
		}
	}
	return defaultValue
}