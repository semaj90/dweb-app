/**
 * Production Configuration Management
 * Optimized for memory usage, caching, and Windows native performance
 */

package config

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"sync"

	"github.com/joho/godotenv"
)

// ProductionConfig holds all production configuration
type ProductionConfig struct {
	// Service Configuration
	ServiceName        string `json:"service_name"`
	ServiceDescription string `json:"service_description"`
	Port               int    `json:"port"`
	Host               string `json:"host"`
	
	// GPU Acceleration Settings
	GPU struct {
		Enabled       bool   `json:"enabled"`
		DeviceID      int    `json:"device_id"`
		MemoryLimit   int64  `json:"memory_limit_mb"`
		BatchSize     int    `json:"batch_size"`
		CUDAVersion   string `json:"cuda_version"`
		Optimization  string `json:"optimization"` // "memory", "performance", "balanced"
	} `json:"gpu"`
	
	// Memory Optimization
	Memory struct {
		MaxHeapSize     int64 `json:"max_heap_size_mb"`
		GCTarget        int   `json:"gc_target_percent"`
		EnableProfiling bool  `json:"enable_profiling"`
		PoolSize        int   `json:"pool_size"`
		BufferSize      int   `json:"buffer_size"`
	} `json:"memory"`
	
	// Caching Configuration
	Cache struct {
		Redis struct {
			Enabled      bool   `json:"enabled"`
			Host         string `json:"host"`
			Port         int    `json:"port"`
			Password     string `json:"password"`
			DB           int    `json:"db"`
			MaxRetries   int    `json:"max_retries"`
			PoolSize     int    `json:"pool_size"`
			DialTimeout  int    `json:"dial_timeout_ms"`
			ReadTimeout  int    `json:"read_timeout_ms"`
			WriteTimeout int    `json:"write_timeout_ms"`
		} `json:"redis"`
		
		Local struct {
			Enabled     bool  `json:"enabled"`
			MaxSize     int64 `json:"max_size_mb"`
			TTL         int   `json:"ttl_seconds"`
			CleanupFreq int   `json:"cleanup_frequency_seconds"`
		} `json:"local"`
		
		L2 struct {
			Enabled    bool   `json:"enabled"`
			Type       string `json:"type"` // "loki", "badger", "bbolt"
			Path       string `json:"path"`
			MaxSize    int64  `json:"max_size_mb"`
			Compressed bool   `json:"compressed"`
		} `json:"l2"`
	} `json:"cache"`
	
	// Database Configuration
	Database struct {
		PostgreSQL struct {
			Host            string `json:"host"`
			Port            int    `json:"port"`
			User            string `json:"user"`
			Password        string `json:"password"`
			Database        string `json:"database"`
			SSLMode         string `json:"ssl_mode"`
			MaxOpenConns    int    `json:"max_open_connections"`
			MaxIdleConns    int    `json:"max_idle_connections"`
			ConnMaxLifetime int    `json:"connection_max_lifetime_minutes"`
		} `json:"postgresql"`
		
		Neo4j struct {
			URI      string `json:"uri"`
			Username string `json:"username"`
			Password string `json:"password"`
			MaxConns int    `json:"max_connections"`
		} `json:"neo4j"`
	} `json:"database"`
	
	// Logging Configuration
	Logging struct {
		Level       string `json:"level"`
		Format      string `json:"format"` // "json", "text"
		OutputPath  string `json:"output_path"`
		MaxSize     int    `json:"max_size_mb"`
		MaxBackups  int    `json:"max_backups"`
		MaxAge      int    `json:"max_age_days"`
		Compress    bool   `json:"compress"`
		EventLog    bool   `json:"windows_event_log"`
	} `json:"logging"`
	
	// Performance Tuning
	Performance struct {
		WorkerCount       int  `json:"worker_count"`
		MaxConcurrency    int  `json:"max_concurrency"`
		EnablePipelining  bool `json:"enable_pipelining"`
		BatchTimeout      int  `json:"batch_timeout_ms"`
		KeepAlive         bool `json:"keep_alive"`
		ReadBufferSize    int  `json:"read_buffer_size"`
		WriteBufferSize   int  `json:"write_buffer_size"`
		EnableCompression bool `json:"enable_compression"`
	} `json:"performance"`
	
	// Windows-specific settings
	Windows struct {
		ServiceAccount    string `json:"service_account"`
		ServiceStartType  string `json:"service_start_type"` // "auto", "manual", "disabled"
		EventLogSource    string `json:"event_log_source"`
		ProcessPriority   string `json:"process_priority"`   // "idle", "normal", "high", "realtime"
		AffinityMask      uint64 `json:"affinity_mask"`      // CPU affinity bitmask
		WorkingSetLimit   int64  `json:"working_set_limit_mb"`
		PrivateMemoryLimit int64 `json:"private_memory_limit_mb"`
	} `json:"windows"`
}

var (
	globalConfig *ProductionConfig
	configMutex  sync.RWMutex
	configCache  = make(map[string]interface{})
)

// LoadProductionConfig loads optimized production configuration
func LoadProductionConfig(configPath string) (*ProductionConfig, error) {
	configMutex.Lock()
	defer configMutex.Unlock()
	
	if globalConfig != nil {
		return globalConfig, nil
	}
	
	config := &ProductionConfig{}
	
	// Load environment variables first
	if err := godotenv.Load(); err != nil {
		// Don't error if .env doesn't exist in production
		fmt.Printf("Warning: Could not load .env file: %v\n", err)
	}
	
	// Set production defaults optimized for Windows and GPU acceleration
	config.setProductionDefaults()
	
	// Load from JSON config file if provided
	if configPath != "" {
		if err := config.loadFromFile(configPath); err != nil {
			return nil, fmt.Errorf("failed to load config file: %v", err)
		}
	}
	
	// Override with environment variables
	config.loadFromEnvironment()
	
	// Apply memory optimizations
	config.applyMemoryOptimizations()
	
	// Validate configuration
	if err := config.validate(); err != nil {
		return nil, fmt.Errorf("invalid configuration: %v", err)
	}
	
	globalConfig = config
	return config, nil
}

// setProductionDefaults sets optimized defaults for Windows production
func (c *ProductionConfig) setProductionDefaults() {
	// Service defaults
	c.ServiceName = "LegalAI-Service"
	c.ServiceDescription = "Legal AI Platform Microservice"
	c.Port = 8080
	c.Host = "0.0.0.0"
	
	// GPU defaults optimized for RTX 3060 Ti
	c.GPU.Enabled = true
	c.GPU.DeviceID = 0
	c.GPU.MemoryLimit = 6144 // 6GB for RTX 3060 Ti
	c.GPU.BatchSize = 32
	c.GPU.CUDAVersion = "11.8"
	c.GPU.Optimization = "balanced"
	
	// Memory defaults optimized for production
	c.Memory.MaxHeapSize = 4096 // 4GB
	c.Memory.GCTarget = 100     // Default Go GC target
	c.Memory.EnableProfiling = false
	c.Memory.PoolSize = runtime.NumCPU() * 2
	c.Memory.BufferSize = 8192
	
	// Cache defaults with multi-tier caching
	c.Cache.Redis.Enabled = true
	c.Cache.Redis.Host = "localhost"
	c.Cache.Redis.Port = 6379
	c.Cache.Redis.DB = 0
	c.Cache.Redis.MaxRetries = 3
	c.Cache.Redis.PoolSize = runtime.NumCPU() * 2
	c.Cache.Redis.DialTimeout = 5000
	c.Cache.Redis.ReadTimeout = 3000
	c.Cache.Redis.WriteTimeout = 3000
	
	c.Cache.Local.Enabled = true
	c.Cache.Local.MaxSize = 512 // 512MB
	c.Cache.Local.TTL = 3600    // 1 hour
	c.Cache.Local.CleanupFreq = 300 // 5 minutes
	
	c.Cache.L2.Enabled = true
	c.Cache.L2.Type = "badger"
	c.Cache.L2.Path = "./cache"
	c.Cache.L2.MaxSize = 1024 // 1GB
	c.Cache.L2.Compressed = true
	
	// Database defaults
	c.Database.PostgreSQL.Host = "localhost"
	c.Database.PostgreSQL.Port = 5432
	c.Database.PostgreSQL.Database = "legal_ai_db"
	c.Database.PostgreSQL.SSLMode = "disable"
	c.Database.PostgreSQL.MaxOpenConns = 25
	c.Database.PostgreSQL.MaxIdleConns = 5
	c.Database.PostgreSQL.ConnMaxLifetime = 60
	
	c.Database.Neo4j.URI = "bolt://localhost:7687"
	c.Database.Neo4j.Username = "neo4j"
	c.Database.Neo4j.MaxConns = 10
	
	// Logging defaults
	c.Logging.Level = "info"
	c.Logging.Format = "json"
	c.Logging.OutputPath = "./logs/service.log"
	c.Logging.MaxSize = 100
	c.Logging.MaxBackups = 3
	c.Logging.MaxAge = 30
	c.Logging.Compress = true
	c.Logging.EventLog = true
	
	// Performance defaults optimized for production
	c.Performance.WorkerCount = runtime.NumCPU()
	c.Performance.MaxConcurrency = runtime.NumCPU() * 4
	c.Performance.EnablePipelining = true
	c.Performance.BatchTimeout = 100
	c.Performance.KeepAlive = true
	c.Performance.ReadBufferSize = 32768
	c.Performance.WriteBufferSize = 32768
	c.Performance.EnableCompression = true
	
	// Windows-specific defaults
	c.Windows.ServiceAccount = `NT AUTHORITY\LocalService`
	c.Windows.ServiceStartType = "auto"
	c.Windows.EventLogSource = c.ServiceName
	c.Windows.ProcessPriority = "normal"
	c.Windows.AffinityMask = 0 // Use all CPUs
	c.Windows.WorkingSetLimit = 8192 // 8GB
	c.Windows.PrivateMemoryLimit = 6144 // 6GB
}

// loadFromFile loads configuration from JSON file
func (c *ProductionConfig) loadFromFile(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	
	return json.Unmarshal(data, c)
}

// loadFromEnvironment overrides config with environment variables
func (c *ProductionConfig) loadFromEnvironment() {
	// Service configuration
	if val := os.Getenv("SERVICE_NAME"); val != "" {
		c.ServiceName = val
	}
	if val := os.Getenv("SERVICE_DESCRIPTION"); val != "" {
		c.ServiceDescription = val
	}
	if val := os.Getenv("PORT"); val != "" {
		if port, err := strconv.Atoi(val); err == nil {
			c.Port = port
		}
	}
	if val := os.Getenv("HOST"); val != "" {
		c.Host = val
	}
	
	// GPU configuration
	if val := os.Getenv("GPU_ENABLED"); val != "" {
		c.GPU.Enabled = val == "true"
	}
	if val := os.Getenv("GPU_DEVICE_ID"); val != "" {
		if id, err := strconv.Atoi(val); err == nil {
			c.GPU.DeviceID = id
		}
	}
	if val := os.Getenv("GPU_MEMORY_LIMIT_MB"); val != "" {
		if limit, err := strconv.ParseInt(val, 10, 64); err == nil {
			c.GPU.MemoryLimit = limit
		}
	}
	if val := os.Getenv("GPU_BATCH_SIZE"); val != "" {
		if size, err := strconv.Atoi(val); err == nil {
			c.GPU.BatchSize = size
		}
	}
	if val := os.Getenv("GPU_OPTIMIZATION"); val != "" {
		c.GPU.Optimization = val
	}
	
	// Memory configuration
	if val := os.Getenv("MAX_HEAP_SIZE_MB"); val != "" {
		if size, err := strconv.ParseInt(val, 10, 64); err == nil {
			c.Memory.MaxHeapSize = size
		}
	}
	if val := os.Getenv("GC_TARGET_PERCENT"); val != "" {
		if target, err := strconv.Atoi(val); err == nil {
			c.Memory.GCTarget = target
		}
	}
	
	// Database configuration
	if val := os.Getenv("DATABASE_HOST"); val != "" {
		c.Database.PostgreSQL.Host = val
	}
	if val := os.Getenv("DATABASE_PORT"); val != "" {
		if port, err := strconv.Atoi(val); err == nil {
			c.Database.PostgreSQL.Port = port
		}
	}
	if val := os.Getenv("DATABASE_USER"); val != "" {
		c.Database.PostgreSQL.User = val
	}
	if val := os.Getenv("DATABASE_PASSWORD"); val != "" {
		c.Database.PostgreSQL.Password = val
	}
	if val := os.Getenv("DATABASE_NAME"); val != "" {
		c.Database.PostgreSQL.Database = val
	}
	
	// Redis configuration
	if val := os.Getenv("REDIS_HOST"); val != "" {
		c.Cache.Redis.Host = val
	}
	if val := os.Getenv("REDIS_PORT"); val != "" {
		if port, err := strconv.Atoi(val); err == nil {
			c.Cache.Redis.Port = port
		}
	}
	if val := os.Getenv("REDIS_PASSWORD"); val != "" {
		c.Cache.Redis.Password = val
	}
	
	// Neo4j configuration
	if val := os.Getenv("NEO4J_URI"); val != "" {
		c.Database.Neo4j.URI = val
	}
	if val := os.Getenv("NEO4J_USERNAME"); val != "" {
		c.Database.Neo4j.Username = val
	}
	if val := os.Getenv("NEO4J_PASSWORD"); val != "" {
		c.Database.Neo4j.Password = val
	}
}

// applyMemoryOptimizations configures Go runtime for optimal memory usage
func (c *ProductionConfig) applyMemoryOptimizations() {
	// Set GOGC environment variable
	if c.Memory.GCTarget != 100 {
		os.Setenv("GOGC", strconv.Itoa(c.Memory.GCTarget))
	}
	
	// Set GOMEMLIMIT if specified
	if c.Memory.MaxHeapSize > 0 {
		limit := fmt.Sprintf("%dMiB", c.Memory.MaxHeapSize)
		os.Setenv("GOMEMLIMIT", limit)
	}
	
	// Set GOMAXPROCS based on worker count
	if c.Performance.WorkerCount > 0 {
		runtime.GOMAXPROCS(c.Performance.WorkerCount)
	}
}

// validate ensures configuration is valid
func (c *ProductionConfig) validate() error {
	if c.Port < 1 || c.Port > 65535 {
		return fmt.Errorf("invalid port: %d", c.Port)
	}
	
	if c.GPU.Enabled {
		if c.GPU.DeviceID < 0 {
			return fmt.Errorf("invalid GPU device ID: %d", c.GPU.DeviceID)
		}
		if c.GPU.MemoryLimit <= 0 {
			return fmt.Errorf("invalid GPU memory limit: %d", c.GPU.MemoryLimit)
		}
		if c.GPU.BatchSize <= 0 {
			return fmt.Errorf("invalid GPU batch size: %d", c.GPU.BatchSize)
		}
	}
	
	if c.Memory.MaxHeapSize <= 0 {
		return fmt.Errorf("invalid max heap size: %d", c.Memory.MaxHeapSize)
	}
	
	if c.Performance.WorkerCount <= 0 {
		return fmt.Errorf("invalid worker count: %d", c.Performance.WorkerCount)
	}
	
	return nil
}

// SaveToFile saves the current configuration to a JSON file
func (c *ProductionConfig) SaveToFile(path string) error {
	// Create directory if it doesn't exist
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}
	
	data, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return err
	}
	
	return os.WriteFile(path, data, 0644)
}

// GetCachedValue returns a cached configuration value
func GetCachedValue(key string) (interface{}, bool) {
	configMutex.RLock()
	defer configMutex.RUnlock()
	
	value, exists := configCache[key]
	return value, exists
}

// SetCachedValue caches a configuration value
func SetCachedValue(key string, value interface{}) {
	configMutex.Lock()
	defer configMutex.Unlock()
	
	configCache[key] = value
}

// GetConfig returns the global configuration instance
func GetConfig() *ProductionConfig {
	configMutex.RLock()
	defer configMutex.RUnlock()
	
	return globalConfig
}

// ReloadConfig reloads configuration from file and environment
func ReloadConfig(configPath string) error {
	configMutex.Lock()
	defer configMutex.Unlock()
	
	globalConfig = nil
	configCache = make(map[string]interface{})
	
	_, err := LoadProductionConfig(configPath)
	return err
}