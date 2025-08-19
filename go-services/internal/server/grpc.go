package server

import (
	"crypto/tls"

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

	// Configure TLS for production
	if c.Grpc.EnableTls {
		tlsConfig := &tls.Config{
			MinVersion: tls.VersionTLS12,
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
	if c.Grpc.Timeout != nil {
		opts = append(opts, grpc.Timeout(c.Grpc.Timeout.AsDuration()))
	}

	srv := grpc.NewServer(opts...)

	// Register legal AI services
	pb.RegisterLegalAnalysisServiceServer(srv, legalSvc)
	pb.RegisterVectorSearchServiceServer(srv, vectorSvc)

	return srv
}

// Note: QUIC functionality moved to quic.go for better organization