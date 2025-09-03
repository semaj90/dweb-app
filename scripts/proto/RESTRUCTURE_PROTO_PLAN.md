# Proto Restructure Plan

This document is auto-generated and will be removed after completion.

Goal: Eliminate mixed Go packages (embed, wire, gpu, tensor, ingest) within single directory `go-microservice/proto/proto` that blocks `go build ./...`.

Approach:
1. Create subfolders: proto/embed, proto/events, proto/gpu, proto/tensor, proto/ingest.
2. Move matching *_pb.go & *_grpc.pb.go files into their respective subfolders.
3. Update module import paths in Go code where protobuf types will be used (currently gateway uses placeholders, so minimal impact now).
4. Add internal umbrella package `protoindex` (optional) providing typed re-exports if convenient.
5. Provide regeneration script stub: `scripts/proto/regenerate-protos.ps1` for future consistency.

Mapping:
- embed.pb.go, embed_grpc.pb.go  -> proto/embed/
- events.pb.go                   -> proto/events/
- gpu_service.pb.go, gpu_service_grpc.pb.go -> proto/gpu/
- tensor.pb.go, tensor_grpc.pb.go -> proto/tensor/
- ingest.pb.go, ingest_grpc.pb.go -> proto/ingest/

Post-steps:
- Verify `go build ./...` passes.
- Remove old `proto/proto` directory if empty; keep README for transition if needed.
- Update README_PROTO_PACKAGES.md noting completion.
