package conf

import (
	"time"

	"google.golang.org/protobuf/types/known/durationpb"
)

// Server configuration
type Server struct {
	Grpc *GRPC
	Http *HTTP
	Quic *QUIC
}

// gRPC server configuration
type GRPC struct {
	Network   string
	Addr      string
	Timeout   *durationpb.Duration
	EnableTls bool
}

// HTTP server configuration
type HTTP struct {
	Network string
	Addr    string
	Timeout *durationpb.Duration
}

// QUIC server configuration
type QUIC struct {
	Network string
	Addr    string
	Timeout *durationpb.Duration
}

// Database configuration
type Database struct {
	Driver string
	Source string
}

// Redis configuration
type Redis struct {
	Network  string
	Addr     string
	Password string
	DB       int
}

// Ollama configuration
type Ollama struct {
	BaseURL string
	Model   string
}

// Bootstrap contains all configuration
type Bootstrap struct {
	Server   *Server
	Database *Database
	Redis    *Redis
	Ollama   *Ollama
}

// NewBootstrap creates default configuration
func NewBootstrap() *Bootstrap {
	return &Bootstrap{
		Server: &Server{
			Grpc: &GRPC{
				Network:   "tcp",
				Addr:      ":8084",
				Timeout:   durationpb.New(time.Second * 30),
				EnableTls: false,
			},
			Http: &HTTP{
				Network: "tcp",
				Addr:    ":8083",
				Timeout: durationpb.New(time.Second * 30),
			},
			Quic: &QUIC{
				Network: "udp",
				Addr:    ":8443",
				Timeout: durationpb.New(time.Second * 30),
			},
		},
		Database: &Database{
			Driver: "postgres",
			Source: "postgres://legal_admin:123456@localhost:5432/legal_ai_db?sslmode=disable",
		},
		Redis: &Redis{
			Network:  "tcp",
			Addr:     "localhost:6379",
			Password: "",
			DB:       0,
		},
		Ollama: &Ollama{
			BaseURL: "http://localhost:11434",
			Model:   "gemma3-legal",
		},
	}
}
