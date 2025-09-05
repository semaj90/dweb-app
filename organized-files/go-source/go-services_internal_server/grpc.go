package server

import (
	"context"
	"crypto/tls"
	"time"

	"github.com/go-kratos/kratos/v2/log"
	"github.com/go-kratos/kratos/v2/middleware/logging"
	"github.com/go-kratos/kratos/v2/middleware/metrics"
	"github.com/go-kratos/kratos/v2/middleware/recovery"
	"github.com/go-kratos/kratos/v2/middleware/tracing"
	"github.com/go-kratos/kratos/v2/transport/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/keepalive"

	pb "../../api/legal/v1"
	"../conf"
	"../service"
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
		creds := credentials.NewTLS(tlsConfig)
		opts = append(opts, grpc.TLSConfig(creds))
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

	// Configure keep-alive settings for long-running connections
	opts = append(opts, grpc.Options(
		grpc.KeepaliveParams(keepalive.ServerParameters{
			Time:    30 * time.Second,
			Timeout: 5 * time.Second,
		}),
		grpc.KeepaliveEnforcementPolicy(keepalive.EnforcementPolicy{
			MinTime:             10 * time.Second,
			PermitWithoutStream: true,
		}),
	))

	srv := grpc.NewServer(opts...)

	// Register legal AI services
	pb.RegisterLegalAnalysisServiceServer(srv, legalSvc)
	pb.RegisterVectorSearchServiceServer(srv, vectorSvc)

	return srv
}

// QUIC Server for ultra-low latency communication
func NewQUICServer(
	c *conf.Server,
	legalSvc *service.LegalService,
	logger log.Logger,
) *quic.Server {
	// QUIC configuration for legal AI real-time processing
	config := &quic.Config{
		MaxIdleTimeout:        30 * time.Second,
		MaxIncomingStreams:    100,
		MaxIncomingUniStreams: 100,
		KeepAlive:            true,
	}

	server := &quic.Server{
		Addr:    c.Quic.Addr,
		Config:  config,
		Handler: NewQUICHandler(legalSvc, logger),
	}

	return server
}

// QUIC Handler for legal document processing
type QUICHandler struct {
	legalSvc *service.LegalService
	logger   log.Logger
}

func NewQUICHandler(legalSvc *service.LegalService, logger log.Logger) *QUICHandler {
	return &QUICHandler{
		legalSvc: legalSvc,
		logger:   logger,
	}
}

func (h *QUICHandler) HandleStream(ctx context.Context, stream quic.Stream) {
	defer stream.Close()

	// Read document data from QUIC stream
	buffer := make([]byte, 64*1024) // 64KB buffer
	n, err := stream.Read(buffer)
	if err != nil {
		h.logger.Errorf("QUIC stream read error: %v", err)
		return
	}

	// Process document with legal AI pipeline
	result, err := h.legalSvc.ProcessDocumentStream(ctx, buffer[:n])
	if err != nil {
		h.logger.Errorf("Document processing error: %v", err)
		return
	}

	// Send results back over QUIC stream
	_, err = stream.Write(result)
	if err != nil {
		h.logger.Errorf("QUIC stream write error: %v", err)
	}
}