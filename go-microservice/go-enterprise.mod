module vector-consumer-enterprise

go 1.21

require (
	// Core gRPC and protobuf
	google.golang.org/grpc v1.58.3
	google.golang.org/protobuf v1.31.0
	
	// Enterprise database layer
	github.com/jackc/pgx/v5 v5.4.3
	github.com/golang-migrate/migrate/v4 v4.16.2
	
	// High-performance caching
	github.com/go-redis/redis/v8 v8.11.5
	github.com/dgraph-io/ristretto v0.1.1
	
	// Message queuing
	github.com/streadway/amqp v1.1.0
	
	// Observability and metrics
	github.com/prometheus/client_golang v1.17.0
	github.com/sirupsen/logrus v1.9.3
	go.opentelemetry.io/otel v1.19.0
	go.opentelemetry.io/otel/trace v1.19.0
	
	// Kratos integration (placeholder)
	github.com/ory/kratos-client-go v0.13.1
	
	// JSON handling
	github.com/tidwall/gjson v1.17.0
	github.com/tidwall/sjson v1.2.5
)

require (
	// Transitive dependencies (major ones)
	github.com/jackc/pgpassfile v1.0.0 // indirect
	github.com/jackc/pgservicefile v0.0.0-20221227161230-091c0ba34f0a // indirect
	golang.org/x/crypto v0.12.0 // indirect
	golang.org/x/net v0.14.0 // indirect
	golang.org/x/sys v0.11.0 // indirect
	golang.org/x/text v0.12.0 // indirect
	google.golang.org/genproto/googleapis/rpc v0.0.0-20230822172742-b8732ec3820d // indirect
)