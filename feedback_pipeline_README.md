Feedback pipeline (quick start)

Files added:
- go-microservice/proto/feedback.proto  — protobuf schema for feedback events
- go-microservice/cmd/collector/main.go  — example collector that writes to Postgres + Redis stream
- go-microservice/cmd/aggregator/main.go — example aggregator consumer that batches and upserts to Qdrant
- scripts/ml/ae_train.py  — PyTorch autoencoder training example
- scripts/ml/cluster_upsert.py — k-means clustering + Postgres tag upsert example

Quick steps:
1. Generate Go proto bindings (repo-local protoc recommended):
   - install protoc + protoc-gen-go + protoc-gen-go-grpc
   - protoc --proto_path=go-microservice/proto --go_out=.. --go-grpc_out=.. go-microservice/proto/feedback.proto

2. Ensure Postgres and Redis are running. Create the table (example using psql):
   CREATE EXTENSION IF NOT EXISTS vector;
   CREATE TABLE feedback_events (
     id uuid PRIMARY KEY,
     user_id text,
     job_id text,
     ts timestamptz,
     action text,
     reward double precision,
     meta jsonb,
     vec vector(768),
     proto bytea
   );

3. Build and run collector:
   cd go-microservice/cmd/collector
   go build -o collector.exe
   ./collector.exe

4. Run aggregator after proto is generated and collector is producing to Redis.

5. Use scripts in scripts/ml to train autoencoder and cluster embeddings.

Notes:
- The Go examples are illustrative; adjust import paths to match generated proto package location and module name.
- For production: consider batching, retries, monitoring, and backpressure on Redis/Qdrant.
