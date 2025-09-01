import { json, type RequestHandler } from '@sveltejs/kit';
import { reinforcementLearningCache } from '$lib/caching/reinforcement-learning-cache.server';
import { multidimensionalRoutingMatrix } from '$lib/routing/multidimensional-routing-matrix.server';
import { physicsAwareGPUOrchestrator } from '$lib/gpu/physics-aware-gpu-orchestrator.server';
import { buildSuccessResponse, buildErrorResponse } from '$lib/server/api/response';
import { buildCognitiveMetrics, deriveEmergentCognitiveSignals } from '$lib/types/metrics';

async function ensureInitialized() {
  await Promise.all([
    reinforcementLearningCache.initialize(),
    multidimensionalRoutingMatrix.initialize(),
    physicsAwareGPUOrchestrator.initialize()
  ]);
}

export const GET: RequestHandler = async ({ locals }) => {
  const start = performance.now();
  try {
    await ensureInitialized();
    let metrics = buildCognitiveMetrics({
      routingEfficiency: multidimensionalRoutingMatrix.getEfficiencyScore() * 100,
      cacheHitRatio: reinforcementLearningCache.getHitRatio() * 100,
      gpuUtilization: physicsAwareGPUOrchestrator.getGPUUtilization() * 100
    });
    metrics = deriveEmergentCognitiveSignals(metrics);

    const processingTimeMs = performance.now() - start;
    const requestId = (locals as any)?.requestId || `metrics_${Date.now()}`;
    const body = buildSuccessResponse(metrics, { processingTimeMs, requestId });
    const response = json(body, { status: 200 });
    response.headers.set('X-Metrics-Snapshot', 'true');
    response.headers.set('Cache-Control', 'no-store');
    return response;
  } catch (err: any) {
    const processingTimeMs = performance.now() - start;
    const requestId = (locals as any)?.requestId || `metrics_${Date.now()}`;
    const body = buildErrorResponse('METRICS_ERROR', err?.message || 'Failed to gather metrics', { processingTimeMs, requestId });
    const response = json(body, { status: 500 });
    response.headers.set('X-Metrics-Snapshot', 'false');
    return response;
  }
};
