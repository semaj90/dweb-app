package server

import (
	"crypto/tls"
	"time"

	"github.com/go-kratos/kratos/v2/log"
	"github.com/go-kratos/kratos/v2/middleware/logging"
	"github.com/go-kratos/kratos/v2/middleware/metrics"
	"github.com/go-kratos/kratos/v2/middleware/recovery"
	"github.com/go-kratos/kratos/v2/middleware/tracing"
	"github.com/go-kratos/kratos/v2/transport/grpc"

	pb "legal-ai-services/api/legal/v1"
	"legal-ai-services/internal/conf"
	"legal-ai-services/internal/service"
)

// NewGRPCServer creates a new gRPC server with legal AI services
func NewGRPCServer(
	c *conf.Server,
	legalSvc *service.LegalService,
	vectorSvc *service.VectorService,
	logger log.Logger,
) *grpc.Server {
	var opts = []grpc.ServerOption{
		grpc.Middleware(
			recovery.Recovery(),
			tracing.Server(),
			logging.Server(logger),
			metrics.Server(),
		),
	}

	// Configure TLS 1.3 for production with proper cipher suites
	if c.Grpc.EnableTls {
		tlsConfig := &tls.Config{
			MinVersion: tls.VersionTLS13,
			MaxVersion: tls.VersionTLS13,
			CipherSuites: []uint16{
				tls.TLS_AES_256_GCM_SHA384,
				tls.TLS_CHACHA20_POLY1305_SHA256,
				tls.TLS_AES_128_GCM_SHA256,
			},
			CurvePreferences: []tls.CurveID{
				tls.X25519,
				tls.CurveP384,
				tls.CurveP256,
			},
			PreferServerCipherSuites: true,
		}
		opts = append(opts, grpc.TLSConfig(tlsConfig))
	}

	// Configure network settings
	if c.Grpc.Network != "" {
		opts = append(opts, grpc.Network(c.Grpc.Network))
	}
	if c.Grpc.Addr != "" {
		opts = append(opts, grpc.Address(c.Grpc.Addr))
	}
	
	// Add proper timeout handling
	if c.Grpc.Timeout != nil {
		opts = append(opts, grpc.Timeout(c.Grpc.Timeout.AsDuration()))
	} else {
		// Set default timeout of 30 seconds
		opts = append(opts, grpc.Timeout(30*time.Second))
	}

	// Message size limits are set via server options in newer gRPC versions

	srv := grpc.NewServer(opts...)

	// Register legal AI services
	pb.RegisterLegalAnalysisServiceServer(srv, legalSvc)
	pb.RegisterVectorSearchServiceServer(srv, vectorSvc)

	return srv
}

// Note: QUIC functionality moved to quic.go for better organization