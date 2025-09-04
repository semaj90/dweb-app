/**
 * Neo4j Planner Singleton (Server-Friendly)
 * Provides a lazily initialized AlphaGo-style planner without requiring a real WebGL/WebGPU context.
 * If running server-side (no DOM), we supply a stub visualizer that returns empty image data.
 */
import { Neo4jAlphaGoPlanner, type Neo4jPlannerConfig } from './neo4j-alphago-planner'
import { defaultVisualizationConfig, GraphVisualizationEngine } from './graph-visualization-engine';
import { MultiLayerCache } from './multi-layer-cache';

let planner: Neo4jAlphaGoPlanner | null = null;
let initialized = false;

function createStubVisualizer(): GraphVisualizationEngine {
  // Cast minimal stub to expected type; methods not used in headless mode.
  return {
    generateGraphVisualization: async () => '',
    cleanup: async () => {},
  } as any as GraphVisualizationEngine;
}

export async function getPlanner(config: Partial<Neo4jPlannerConfig> = {}): Promise<Neo4jAlphaGoPlanner> {
  if (planner && initialized) return planner;

  const baseConfig: Neo4jPlannerConfig = {
    neo4jUrl: 'bolt://localhost:7687',
    username: 'neo4j',
    password: 'password',
    database: 'neo4j',
    maxDepth: 5,
    evaluationDepth: 3,
    mctsIterations: 250,
    explorationConstant: 1.4,
    enableGPUAcceleration: false,
    ...config
  } as Neo4jPlannerConfig;

  // Attempt to create a real visualizer only if DOM available
  let visualizer: GraphVisualizationEngine;
  try {
    if (typeof document !== 'undefined') {
      visualizer = new GraphVisualizationEngine({ ...defaultVisualizationConfig }, new MultiLayerCache({
        enableRedisCache: false,
        enableLokiCache: false,
        enableMemoryCache: true,
        redisTTL: 0,
        lokiTTL: 0,
        memoryTTL: 300,
        maxMemorySize: 50_000_000,
        maxLokiSize: 0,
        compressionEnabled: false,
        encryptionEnabled: false
      } as any));
      await visualizer.initialize?.().catch(()=>{});
    } else {
      visualizer = createStubVisualizer();
    }
  } catch {
    visualizer = createStubVisualizer();
  }

  planner = new Neo4jAlphaGoPlanner(baseConfig, visualizer);
  await planner.initializeNeo4jConnection();
  initialized = true;
  return planner;
}
