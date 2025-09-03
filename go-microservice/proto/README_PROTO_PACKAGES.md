Proto Package Note
===================

Current generated protobuf Go files in proto/proto use multiple package names:
- embed (embed.pb.go, embed_grpc.pb.go)
- wire  (events.pb.go)
- gpu   (gpu_service.pb.go, gpu_service_grpc.pb.go)
- tensor (tensor.pb.go, tensor_grpc.pb.go)
- ingest (ingest.pb.go, ingest_grpc.pb.go)

Go module path aggregation (importing proto/proto) fails with: found packages embed and wire ...

Short-Term Strategy:
1. Keep gateway code using temporary placeholder types until regeneration.
2. Create per-package subdirectories during regeneration (e.g., proto/embed, proto/events, proto/gpu, proto/tensor, proto/ingest) to avoid mixed packages in one directory.
3. Update imports to pbembed, pbevents, pbgpu, pbtensor, pbingest aliases.

Action Needed:
Regenerate protos with option: --go_opt=module=legal-ai-production/proto and place outputs into segregated folders.

This file documents the intentional interim state.
