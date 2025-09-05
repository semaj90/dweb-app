// Simple Node-side gRPC client helper for server routes
// Assumptions: install @grpc/grpc-js and @grpc/proto-loader in the frontend workspace
// npm i @grpc/grpc-js @grpc/proto-loader
import * as grpc from '@grpc/grpc-js';
import * as protoLoader from '@grpc/proto-loader';

export type PredictRequest = { text: string };
export type PredictResponse = { summary?: string; scores?: Record<string, number> };

export function createLegalBertClient(address = 'localhost:50051', protoPath = 'protos/legal_bert.proto') {
  const packageDefinition = protoLoader.loadSync(protoPath, {
    keepCase: true,
    longs: String,
    enums: String,
    defaults: true,
    oneofs: true
  });

  const loaded: any = grpc.loadPackageDefinition(packageDefinition);
  // proto layout is assumed to use package `legalbert` with service `LegalBert`
  const service = loaded.legalbert && (loaded.legalbert.LegalBert || loaded.legalbert.LegalBertService);
  if (!service) throw new Error(`Could not find LegalBert service in proto: ${protoPath}`);

  return new service(address, grpc.credentials.createInsecure());
}

export async function predictLegalBert(text: string, opts?: { address?: string; protoPath?: string }): Promise<PredictResponse> {
  const client: any = createLegalBertClient(opts?.address, opts?.protoPath);

  return new Promise((resolve, reject) => {
    client.Predict({ text }, (err: any, resp: any) => {
      if (err) return reject(err);
      resolve(resp as PredictResponse);
    });
  });
}

// Note: This helper is intended for server-side SvelteKit routes only (Node runtime).
// Browsers cannot talk to raw gRPC without grpc-web / envoy or a gateway.
