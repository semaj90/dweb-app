package conf

import (
	"fmt"
	"os"
	"strconv"
)

// Config holds all configuration for the Legal AI services
type Config struct {
	Server   ServerConfig   `json:"server"`
	Database DatabaseConfig `json:"database"`
	Vector   VectorConfig   `json:"vector"`
	Cache    CacheConfig    `json:"cache"`
	AI       AIConfig       `json:"ai"`
	Security SecurityConfig `json:"security"`
}

// ServerConfig contains server-related configuration
type ServerConfig struct {
	HTTPPort int    `json:"http_port"`
	GRPCPort int    `json:"grpc_port"`
	QUICPort int    `json:"quic_port"`
	Host     string `json:"host"`
	Environment string `json:"environment"`
}

// DatabaseConfig contains database connection settings
type DatabaseConfig struct {
	PostgresURL    string `json:"postgres_url"`
	MaxConnections int    `json:"max_connections"`
	ConnTimeout    int    `json:"conn_timeout"`
}

// VectorConfig contains vector database settings
type VectorConfig struct {
	QdrantURL      string `json:"qdrant_url"`
	CollectionName string `json:"collection_name"`
	VectorSize     int    `json:"vector_size"`
}

// CacheConfig contains Redis cache settings
type CacheConfig struct {
	RedisURL    string `json:"redis_url"`
	TTL         int    `json:"ttl"`
	MaxMemory   string `json:"max_memory"`
}

// AIConfig contains AI model settings
type AIConfig struct {
	OllamaURL     string `json:"ollama_url"`
	EmbedModel    string `json:"embed_model"`
	ChatModel     string `json:"chat_model"`
	MaxTokens     int    `json:"max_tokens"`
}

// SecurityConfig contains security settings
type SecurityConfig struct {
	EnableTLS     bool   `json:"enable_tls"`
	CertFile      string `json:"cert_file"`
	KeyFile       string `json:"key_file"`
	JWTSecret     string `json:"jwt_secret"`
}

// LoadConfig loads configuration from environment variables
func LoadConfig() (*Config, error) {
	config := &Config{
		Server: ServerConfig{
			HTTPPort:    getEnvInt("HTTP_PORT", 8080),
			GRPCPort:    getEnvInt("GRPC_PORT", 50051),
			QUICPort:    getEnvInt("QUIC_PORT", 9443),
			Host:        getEnv("HOST", "0.0.0.0"),
			Environment: getEnv("ENVIRONMENT", "development"),
		},
		Database: DatabaseConfig{
			PostgresURL:    getEnv("DATABASE_URL", "postgresql://legal_admin:123456@localhost:5432/legal_ai_db"),
			MaxConnections: getEnvInt("DB_MAX_CONNECTIONS", 25),
			ConnTimeout:    getEnvInt("DB_CONN_TIMEOUT", 30),
		},
		Vector: VectorConfig{
			QdrantURL:      getEnv("QDRANT_URL", "http://localhost:6333"),
			CollectionName: getEnv("QDRANT_COLLECTION", "legal_documents"),
			VectorSize:     getEnvInt("VECTOR_SIZE", 768),
		},
		Cache: CacheConfig{
			RedisURL:  getEnv("REDIS_URL", "redis://localhost:6379"),
			TTL:       getEnvInt("CACHE_TTL", 3600),
			MaxMemory: getEnv("REDIS_MAX_MEMORY", "256mb"),
		},
		AI: AIConfig{
			OllamaURL:  getEnv("OLLAMA_URL", "http://localhost:11434"),
			EmbedModel: getEnv("EMBED_MODEL", "nomic-embed-text"),
			ChatModel:  getEnv("CHAT_MODEL", "gemma3:latest"),
			MaxTokens:  getEnvInt("MAX_TOKENS", 4096),
		},
		Security: SecurityConfig{
			EnableTLS: getEnvBool("ENABLE_TLS", false),
			CertFile:  getEnv("TLS_CERT_FILE", ""),
			KeyFile:   getEnv("TLS_KEY_FILE", ""),
			JWTSecret: getEnv("JWT_SECRET", ""),
		},
	}

	// Validate required configuration
	if err := validateConfig(config); err != nil {
		return nil, fmt.Errorf("invalid configuration: %w", err)
	}

	return config, nil
}

// validateConfig validates the loaded configuration
func validateConfig(config *Config) error {
	if config.Server.HTTPPort <= 0 || config.Server.HTTPPort > 65535 {
		return fmt.Errorf("invalid HTTP port: %d", config.Server.HTTPPort)
	}
	
	if config.Server.GRPCPort <= 0 || config.Server.GRPCPort > 65535 {
		return fmt.Errorf("invalid gRPC port: %d", config.Server.GRPCPort)
	}
	
	if config.Database.PostgresURL == "" {
		return fmt.Errorf("database URL is required")
	}
	
	if config.AI.OllamaURL == "" {
		return fmt.Errorf("Ollama URL is required")
	}
	
	return nil
}

// Utility functions for environment variable parsing

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getEnvInt(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if intValue, err := strconv.Atoi(value); err == nil {
			return intValue
		}
	}
	return defaultValue
}

func getEnvBool(key string, defaultValue bool) bool {
	if value := os.Getenv(key); value != "" {
		if boolValue, err := strconv.ParseBool(value); err == nil {
			return boolValue
		}
	}
	return defaultValue
}