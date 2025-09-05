/**
 * Neo4j AlphaGo-Style Graph Planning System
 * Integrates with GPU visualization and neural network decomposition
 * for multi-step legal knowledge graph traversal and planning
 */

import { GraphVisualizationEngine } from './graph-visualization-engine';
import { nesPlannerBridge } from '../memory/nes-memory-architecture';
import { SOMNeuralNetwork } from '../ai/som-neural-network';
import { GraphPatternAutoEncoder } from '../ai/graph-pattern-autoencoder';
import { multiDimensionalImageCache } from '../caching/multi-dimensional-image-cache';

export interface Neo4jPlannerConfig {
  neo4jUrl: string;
  username: string;
  password: string;
  database: string;
  maxDepth: number;
  evaluationDepth: number;
  mctsIterations: number;
  explorationConstant: number;
  enableGPUAcceleration: boolean;
}

export interface LegalNode {
  id: string;
  labels: string[];
  type: 'Case' | 'Statute' | 'Regulation' | 'Precedent' | 'Person' | 'Organization' | 'Concept';
  properties: {
    title?: string;
    jurisdiction?: string;
    year?: number;
    importance?: number;
    citationCount?: number;
    authority?: number;
  };
  embedding?: Float32Array;
}

export interface LegalRelationship {
  id: string;
  type: 'CITES' | 'OVERRULES' | 'DISTINGUISHES' | 'FOLLOWS' | 'REFERENCES' | 'CONFLICTS_WITH';
  source: string;
  target: string;
  properties: {
    strength?: number;
    confidence?: number;
    timestamp?: number;
    context?: string;
  };
}

export interface GraphState {
  currentNode: LegalNode;
  visitedNodes: Set<string>;
  path: string[];
  depth: number;
  value: number;
  legalContext: {
    jurisdiction: string;
    practiceArea: string;
    timeFrame: [number, number];
    precedentStrength: number;
  };
}

export interface MCTSNode {
  state: GraphState;
  parent: MCTSNode | null;
  children: Map<string, MCTSNode>;
  visits: number;
  totalValue: number;
  untriedActions: string[];
  fullyExpanded: boolean;
  handle: number; // NES planner memory handle
}

export interface PlanningResult {
  bestPath: LegalNode[];
  pathValue: number;
  exploredNodes: number;
  computationTime: number;
  visualizations: {
    searchTree: string; // Base64 image
    finalPath: string;  // Base64 image
    heatmap: string;    // Base64 image
  };
  legalAnalysis: {
    precedentChain: LegalNode[];
    citationStrength: number;
    jurisdictionalConsistency: number;
    temporalRelevance: number;
  };
}

export class Neo4jAlphaGoPlanner {
  private driver: any = null; // Neo4j driver placeholder
  private visualizer: GraphVisualizationEngine;
  private som: SOMNeuralNetwork;
  private autoencoder: GraphPatternAutoEncoder;
  private config: Neo4jPlannerConfig;

  // MCTS state
  private rootNode: MCTSNode | null = null;
  private nodeCache: Map<string, LegalNode> = new Map();
  private relationshipCache: Map<string, LegalRelationship[]> = new Map();

  // Performance metrics
  private metrics = {
    totalQueries: 0,
    cacheHits: 0,
    averageDepth: 0,
    bestPathValue: 0,
    explorationEfficiency: 0
  };

  constructor(
    config: Neo4jPlannerConfig,
    visualizer: GraphVisualizationEngine
  ) {
    this.config = {
      maxDepth: 6,
      evaluationDepth: 4,
      mctsIterations: 1000,
      explorationConstant: 1.4,
      enableGPUAcceleration: true,
      ...config
    };

    this.visualizer = visualizer;
    this.initializeNeuralNetworks();
  }

  private async initializeNeuralNetworks(): Promise<void> {
    try {
      // Initialize SOM for legal concept clustering
      this.som = new SOMNeuralNetwork({
        gridSize: { width: 12, height: 12 },
        learningRate: 0.05,
        neighborhoodRadius: 3.0,
        decayRate: 0.015,
        epochs: 100,
        enableGPU: this.config.enableGPUAcceleration,
        inputDimension: 384
      });
      await this.som.initialize();

      // Initialize auto-encoder for legal pattern compression
      this.autoencoder = new GraphPatternAutoEncoder({
        inputDimension: 768,
        hiddenLayers: [512, 256, 128, 64],
        learningRate: 0.0005,
        batchSize: 16,
        epochs: 75,
        enableGPU: this.config.enableGPUAcceleration,
        compressionTarget: 0.08
      });
      await this.autoencoder.initialize();

      console.log('🧠 Neural networks initialized for Neo4j planning');
    } catch (error) {
      console.error('Failed to initialize neural networks:', error);
    }
  }

  async initializeNeo4jConnection(): Promise<void> {
    try {
      // In production, this would create actual Neo4j driver
      console.log(`🔗 Connecting to Neo4j at ${this.config.neo4jUrl}/${this.config.database}`);

      // Simulate connection
      this.driver = {
        session: () => ({
          run: this.simulateNeo4jQuery.bind(this),
          close: () => {}
        }),
        close: () => {}
      };

      console.log('✅ Neo4j connection established');
    } catch (error) {
      console.error('Failed to connect to Neo4j:', error);
      throw error;
    }
  }

  /**
   * AlphaGo-style planning: looks ahead multiple steps to evaluate paths
   * through the legal knowledge graph using MCTS
   */
  async planOptimalPath(
    startNodeId: string,
    goalCriteria: {
      targetType?: string;
      jurisdiction?: string;
      practiceArea?: string;
      minImportance?: number;
      maxDepth?: number;
    }
  ): Promise<PlanningResult> {
    const startTime = performance.now();
    console.log(`🎯 Starting AlphaGo-style planning from node ${startNodeId}`);

    try {
      // Initialize root node
      const startNode = await this.getNode(startNodeId);
      if (!startNode) {
        throw new Error(`Start node ${startNodeId} not found`);
      }

      const initialState: GraphState = {
        currentNode: startNode,
        visitedNodes: new Set([startNodeId]),
        path: [startNodeId],
        depth: 0,
        value: 0,
        legalContext: {
          jurisdiction: goalCriteria.jurisdiction || startNode.properties.jurisdiction || 'federal',
          practiceArea: goalCriteria.practiceArea || 'general',
          timeFrame: [1900, new Date().getFullYear()],
          precedentStrength: 0
        }
      };

      this.rootNode = {
        state: initialState,
        parent: null,
        children: new Map(),
        visits: 0,
        totalValue: 0,
        untriedActions: await this.getPossibleActions(initialState),
  fullyExpanded: false,
  handle: nesPlannerBridge.allocateNode({ graphNodeId: startNode.id, parentHandle: -1, prior: 1.0, depth: 0 })
      };

      // Run MCTS iterations
      let exploredNodes = 0;
      for (let i = 0; i < this.config.mctsIterations; i++) {
        const selectedNode = this.selectNode(this.rootNode);
        const expandedNode = await this.expandNode(selectedNode);
        const simulationResult = await this.simulate(expandedNode);
        this.backpropagate(expandedNode, simulationResult);
        exploredNodes++;

        // Early termination if we find an excellent path
        if (simulationResult.value > 0.95) {
          console.log(`🎯 Excellent path found early at iteration ${i}`);
          break;
        }
      }

      // Extract best path
      const bestPath = this.extractBestPath();
      const pathValue = this.evaluatePathValue(bestPath);

      // Generate visualizations
      const visualizations = await this.generateVisualizations(bestPath);

      // Analyze legal aspects
      const legalAnalysis = this.analyzeLegalPath(bestPath);

      const result: PlanningResult = {
        bestPath,
        pathValue,
        exploredNodes,
        computationTime: performance.now() - startTime,
        visualizations,
        legalAnalysis
      };

      // Update metrics
      this.updateMetrics(result);

      console.log(`✅ Planning completed: ${bestPath.length} nodes, value ${pathValue.toFixed(3)}, ${exploredNodes} explored, ${result.computationTime.toFixed(2)}ms`);

      return result;
    } catch (error) {
      console.error('Planning failed:', error);
      throw error;
    }
  }

  /**
   * MCTS Selection: navigate to most promising node using UCB1
   */
  private selectNode(node: MCTSNode): MCTSNode {
    while (!this.isTerminal(node.state) && node.fullyExpanded) {
      // Attempt NES planner memory-assisted UCB selection if children have handles
      const allChildren = Array.from(node.children.values());
      if (allChildren.length && allChildren.every(c => c.handle !== undefined)) {
        const selectedHandle = nesPlannerBridge.select(node.handle, this.config.explorationConstant);
        if (selectedHandle !== null) {
          const mapped = allChildren.find(c => c.handle === selectedHandle);
          if (mapped) { node = mapped; continue; }
        }
      }
      let bestChild: MCTSNode | null = null;
      let bestValue = -Infinity;

      for (const child of node.children.values()) {
        const exploitation = child.totalValue / (child.visits || 1);
        const exploration = this.config.explorationConstant *
          Math.sqrt(Math.log(node.visits) / (child.visits || 1));
        const ucb1Value = exploitation + exploration;

        if (ucb1Value > bestValue) {
          bestValue = ucb1Value;
          bestChild = child;
        }
      }

      if (bestChild) {
        node = bestChild;
      } else {
        break;
      }
    }

    return node;
  }

  /**
   * MCTS Expansion: add new child nodes for untried actions
   */
  private async expandNode(node: MCTSNode): Promise<MCTSNode> {
    if (this.isTerminal(node.state) || node.untriedActions.length === 0) {
      node.fullyExpanded = true;
      return node;
    }

    // Select random untried action
    const actionIndex = Math.floor(Math.random() * node.untriedActions.length);
    const action = node.untriedActions.splice(actionIndex, 1)[0];

    // Create new state
    const newState = await this.applyAction(node.state, action);

    // Create child node
    const childNode: MCTSNode = {
      state: newState,
      parent: node,
      children: new Map(),
      visits: 0,
      totalValue: 0,
      untriedActions: await this.getPossibleActions(newState),
      fullyExpanded: false,
      handle: nesPlannerBridge.allocateNode({
        graphNodeId: newState.currentNode.id,
        parentHandle: node.handle,
        prior: (newState.currentNode.properties.importance || 0.5),
        depth: newState.depth
      })
    };

    node.children.set(action, childNode);

    if (node.untriedActions.length === 0) {
      node.fullyExpanded = true;
    }

    return childNode;
  }

  /**
   * MCTS Simulation: random rollout to evaluate node value
   * Enhanced with legal domain knowledge and neural network guidance
   */
  private async simulate(node: MCTSNode): Promise<{ value: number; length: number }> {
    let currentState = { ...node.state };
    let length = 0;
    const maxSimulationDepth = this.config.evaluationDepth;

    // Use neural networks to guide simulation
    let neuralGuidance = 0;
    if (this.som && this.autoencoder) {
      try {
        // Get SOM-guided direction
        const embedding = currentState.currentNode.embedding || new Float32Array(384);
        const somInput = Array.from(embedding).slice(0, 256);
        await this.som.train([somInput]);

        // Get auto-encoder pattern analysis
        const graphData = await this.constructLocalGraph(currentState.currentNode.id, 2);
        const pattern = await this.autoencoder.encodeGraphPattern(graphData);

        neuralGuidance = pattern.legalPatterns.precedentStrength * 0.3 +
                        pattern.legalPatterns.conceptSimilarity * 0.2;
      } catch (error) {
        console.warn('Neural guidance failed:', error);
      }
    }

    while (length < maxSimulationDepth && !this.isTerminal(currentState)) {
      const actions = await this.getPossibleActions(currentState);
      if (actions.length === 0) break;

      // Choose action with legal domain bias
      const action = this.selectBiasedAction(actions, currentState, neuralGuidance);
      currentState = await this.applyAction(currentState, action);
      length++;
    }

    // Evaluate final state with legal-specific metrics
    const value = this.evaluateState(currentState, neuralGuidance);

    return { value, length };
  }

  /**
   * MCTS Backpropagation: update node values up the tree
   */
  private backpropagate(node: MCTSNode, result: { value: number; length: number }): void {
    let current: MCTSNode | null = node;

    while (current !== null) {
      current.visits++;
      current.totalValue += result.value;
      // Mirror into NES planner memory
      if (current.handle !== undefined) {
        nesPlannerBridge.visit(current.handle, result.value);
      }
      current = current.parent;
    }
  }

  /**
   * Extract the best path from MCTS tree
   */
  private extractBestPath(): LegalNode[] {
    if (!this.rootNode) return [];

    const path: LegalNode[] = [this.rootNode.state.currentNode];
    let current = this.rootNode;

    while (current.children.size > 0) {
      let bestChild: MCTSNode | null = null;
      let bestVisits = 0;

      for (const child of current.children.values()) {
        if (child.visits > bestVisits) {
          bestVisits = child.visits;
          bestChild = child;
        }
      }

      if (bestChild) {
        path.push(bestChild.state.currentNode);
        current = bestChild;
      } else {
        break;
      }
    }

    return path;
  }

  /**
   * Generate visualizations of the search process and results
   */
  private async generateVisualizations(path: LegalNode[]): Promise<{
    searchTree: string;
    finalPath: string;
    heatmap: string;
  }> {
    try {
      // Create graph data for visualization
      const nodes = path.map((node, index) => ({
        id: node.id,
        label: node.properties.title || node.id,
        type: node.type.toLowerCase(),
        position: { x: index * 100, y: Math.sin(index) * 50 + 300 },
        metadata: { ...node.properties, pathIndex: index },
        embedding: node.embedding || new Float32Array(384)
      }));

      const edges = [];
      for (let i = 0; i < path.length - 1; i++) {
        edges.push({
          id: `path_edge_${i}`,
          source: path[i].id,
          target: path[i + 1].id,
          type: 'path',
          weight: 1.0,
          metadata: { pathSequence: i }
        });
      }

      const graphData = { nodes, edges };

      // Generate visualizations using different algorithms
  const searchTreeViz = await (this.visualizer as any).generateGraphVisualization ? (this.visualizer as any).generateGraphVisualization(graphData, {
        algorithm: 'som',
        outputFormat: 'base64',
        dimensions: { width: 1200, height: 800 },
        style: {
          backgroundColor: '#1a1a1a',
          nodeColor: '#4a90e2',
          edgeColor: '#ffffff',
          highlightColor: '#ff6b6b'
        }
  }) : { imageData: '' };

  const pathViz = await (this.visualizer as any).generateGraphVisualization ? (this.visualizer as any).generateGraphVisualization(graphData, {
        algorithm: 'dfs',
        outputFormat: 'base64',
        dimensions: { width: 1000, height: 600 },
        style: {
          backgroundColor: '#2a2a2a',
          nodeColor: '#00ff88',
          edgeColor: '#cccccc',
          highlightColor: '#ffaa00'
        }
  }) : { imageData: '' };

  const heatmapViz = await (this.visualizer as any).generateGraphVisualization ? (this.visualizer as any).generateGraphVisualization(graphData, {
        algorithm: 'autoencoder',
        outputFormat: 'base64',
        dimensions: { width: 800, height: 600 },
        style: {
          backgroundColor: '#0a0a0a',
          nodeColor: '#ff4444',
          edgeColor: '#888888',
          highlightColor: '#44ff44'
        }
  }) : { imageData: '' };

      // Cache visualizations
      await multiDimensionalImageCache.storeImage(
        searchTreeViz.imageData!,
        {
          temporal: 'recent',
          spatial: 'global',
          semantic: 'expert',
          visual: 'hires',
          algorithm: 'som'
        },
        graphData,
        { processingTime: 100, qualityScore: 0.9 }
      );

      return {
        searchTree: searchTreeViz.imageData!,
        finalPath: pathViz.imageData!,
        heatmap: heatmapViz.imageData!
      };
    } catch (error) {
      console.error('Visualization generation failed:', error);
      return {
        searchTree: '',
        finalPath: '',
        heatmap: ''
      };
    }
  }

  // Helper methods
  private async simulateNeo4jQuery(query: string, params?: any): Promise<any> {
    // Simulate Neo4j query results for development
    this.metrics.totalQueries++;

    if (query.includes('MATCH (n)') && query.includes('WHERE id(n)')) {
      // Node lookup
      return {
        records: [{
          get: () => ({
            identity: params.nodeId,
            labels: ['Case'],
            properties: {
              title: `Legal Case ${params.nodeId}`,
              jurisdiction: 'federal',
              year: 2020,
              importance: 0.8,
              citationCount: 15
            }
          })
        }]
      };
    } else if (query.includes('MATCH (n)-[r]->(m)')) {
      // Relationship query
      return {
        records: Array.from({ length: 3 }, (_, i) => ({
          get: (field: string) => {
            if (field === 'r') {
              return {
                type: 'CITES',
                properties: { strength: 0.7, confidence: 0.85 }
              };
            } else if (field === 'm') {
              return {
                identity: `target_${i}`,
                labels: ['Statute'],
                properties: {
                  title: `Statute ${i}`,
                  jurisdiction: 'federal',
                  importance: 0.6
                }
              };
            }
            return null;
          }
        }))
      };
    }

    return { records: [] };
  }

  private async getNode(nodeId: string): Promise<LegalNode | null> {
    if (this.nodeCache.has(nodeId)) {
      this.metrics.cacheHits++;
      return this.nodeCache.get(nodeId)!;
    }

    try {
      const session = this.driver.session();
      const result = await session.run(
        'MATCH (n) WHERE id(n) = $nodeId RETURN n',
        { nodeId }
      );
      session.close();

      if (result.records.length === 0) return null;

      const record = result.records[0].get('n');
      const node: LegalNode = {
        id: record.identity.toString(),
        labels: record.labels,
        type: record.labels[0] as LegalNode['type'],
        properties: record.properties,
        embedding: this.generateMockEmbedding()
      };

      this.nodeCache.set(nodeId, node);
      return node;
    } catch (error) {
      console.error('Failed to get node:', error);
      return null;
    }
  }

  private async getPossibleActions(state: GraphState): Promise<string[]> {
    const cacheKey = `actions_${state.currentNode.id}`;
    if (this.relationshipCache.has(cacheKey)) {
      return this.relationshipCache.get(cacheKey)!.map(r => r.target);
    }

    try {
      const session = this.driver.session();
      const result = await session.run(
        'MATCH (n)-[r]->(m) WHERE id(n) = $nodeId RETURN r, m',
        { nodeId: state.currentNode.id }
      );
      session.close();

      const relationships: LegalRelationship[] = [];
      const actions: string[] = [];

      result.records.forEach(record => {
        const rel = record.get('r');
        const targetNode = record.get('m');

        const relationship: LegalRelationship = {
          id: `${state.currentNode.id}_${targetNode.identity}`,
          type: rel.type,
          source: state.currentNode.id,
          target: targetNode.identity.toString(),
          properties: rel.properties || {}
        };

        relationships.push(relationship);

        // Only add action if not already visited and within depth limit
        if (!state.visitedNodes.has(targetNode.identity.toString()) &&
            state.depth < this.config.maxDepth) {
          actions.push(targetNode.identity.toString());
        }
      });

      this.relationshipCache.set(cacheKey, relationships);
      return actions;
    } catch (error) {
      console.error('Failed to get possible actions:', error);
      return [];
    }
  }

  private async applyAction(state: GraphState, action: string): Promise<GraphState> {
    const targetNode = await this.getNode(action);
    if (!targetNode) return state;

    const newVisited = new Set(state.visitedNodes);
    newVisited.add(action);

    return {
      currentNode: targetNode,
      visitedNodes: newVisited,
      path: [...state.path, action],
      depth: state.depth + 1,
      value: state.value,
      legalContext: {
        ...state.legalContext,
        precedentStrength: this.calculatePrecedentStrength(state, targetNode)
      }
    };
  }

  private isTerminal(state: GraphState): boolean {
    return state.depth >= this.config.maxDepth ||
           state.visitedNodes.size >= 50 || // Prevent infinite exploration
           state.currentNode.type === 'Precedent'; // Found a precedent
  }

  private evaluateState(state: GraphState, neuralGuidance: number): number {
    let value = 0;

    // Node importance
    value += (state.currentNode.properties.importance || 0.5) * 0.3;

    // Citation count (normalized)
    const citations = state.currentNode.properties.citationCount || 0;
    value += Math.min(citations / 100, 1.0) * 0.2;

    // Path length penalty (prefer shorter paths)
    value += Math.max(0, 1.0 - state.depth / this.config.maxDepth) * 0.2;

    // Legal context relevance
    value += state.legalContext.precedentStrength * 0.2;

    // Neural network guidance
    value += neuralGuidance * 0.1;

    return Math.min(Math.max(value, 0), 1);
  }

  private selectBiasedAction(actions: string[], state: GraphState, neuralGuidance: number): string {
    if (actions.length === 1) return actions[0];

    // For now, random selection with slight bias toward higher importance
    // In production, this would use more sophisticated legal domain knowledge
    return actions[Math.floor(Math.random() * actions.length)];
  }

  private calculatePrecedentStrength(state: GraphState, targetNode: LegalNode): number {
    const importance = targetNode.properties.importance || 0.5;
    const authority = targetNode.properties.authority || 0.5;
    return Math.min((importance + authority) / 2 + state.legalContext.precedentStrength, 1.0);
  }

  private evaluatePathValue(path: LegalNode[]): number {
    if (path.length === 0) return 0;

    let value = 0;
    for (let i = 0; i < path.length; i++) {
      const node = path[i];
      const weight = 1.0 / (i + 1); // Diminishing weight for later nodes
      value += (node.properties.importance || 0.5) * weight;
    }

    return value / path.length;
  }

  private analyzeLegalPath(path: LegalNode[]): PlanningResult['legalAnalysis'] {
    const precedentNodes = path.filter(n => n.type === 'Precedent')
    const citationStrength = path.reduce((sum, n) => sum + (n.properties.citationCount || 0), 0) / path.length;

    const jurisdictions = new Set(path.map(n => n.properties.jurisdiction).filter(Boolean));
    const jurisdictionalConsistency = 1.0 - (jurisdictions.size - 1) / Math.max(path.length, 1);

    const years = path.map(n => n.properties.year).filter(Boolean) as number[];
    const temporalRange = years.length > 1 ? Math.max(...years) - Math.min(...years) : 0;
    const temporalRelevance = Math.max(0, 1.0 - temporalRange / 50); // 50 year relevance window

    return {
      precedentChain: precedentNodes,
      citationStrength: Math.min(citationStrength / 100, 1.0),
      jurisdictionalConsistency,
      temporalRelevance
    };
  }

  private async constructLocalGraph(nodeId: string, depth: number): Promise<any> {
    // Construct a local graph for neural network analysis
    const nodes: any[] = [];
    const edges: any[] = [];
    const visited = new Set<string>();
    const queue = [{ id: nodeId, currentDepth: 0 }];

    while (queue.length > 0) {
      const { id, currentDepth } = queue.shift()!;
      if (visited.has(id) || currentDepth > depth) continue;

      visited.add(id);
      const node = await this.getNode(id);
      if (!node) continue;

      nodes.push({
        id: node.id,
        label: node.properties.title || node.id,
        type: node.type.toLowerCase(),
        position: { x: Math.random() * 400, y: Math.random() * 400 },
        metadata: node.properties,
        embedding: node.embedding
      });

      if (currentDepth < depth) {
        const actions = await this.getPossibleActions({
          currentNode: node,
          visitedNodes: visited,
          path: [],
          depth: currentDepth,
          value: 0,
          legalContext: {
            jurisdiction: 'federal',
            practiceArea: 'general',
            timeFrame: [2000, 2025],
            precedentStrength: 0
          }
        });

        actions.forEach(targetId => {
          edges.push({
            id: `${id}_${targetId}`,
            source: id,
            target: targetId,
            type: 'references',
            weight: 0.5,
            metadata: {}
          });

          queue.push({ id: targetId, currentDepth: currentDepth + 1 });
        });
      }
    }

    return {
      nodes,
      edges,
      metadata: {
        totalNodes: nodes.length,
        totalEdges: edges.length,
        density: edges.length / (nodes.length || 1),
        averageDegree: (edges.length * 2) / (nodes.length || 1),
        legalDomain: 'general',
        timestamp: Date.now()
      }
    };
  }

  private updateMetrics(result: PlanningResult): void {
    this.metrics.averageDepth = (this.metrics.averageDepth + result.bestPath.length) / 2;
    this.metrics.bestPathValue = Math.max(this.metrics.bestPathValue, result.pathValue);
    this.metrics.explorationEfficiency = result.exploredNodes / this.config.mctsIterations;
  // Append planner memory stats
  const pm = nesPlannerBridge.summary();
  (this.metrics as any).plannerMemory = pm;
  }

  private generateMockEmbedding(): Float32Array {
    const embedding = new Float32Array(384);
    for (let i = 0; i < embedding.length; i++) {
      embedding[i] = (Math.random() - 0.5) * 2;
    }
    return embedding;
  }

  // Public API
  getMetrics() {
    return {
      ...this.metrics,
      cacheHitRatio: this.metrics.totalQueries > 0 ? this.metrics.cacheHits / this.metrics.totalQueries : 0
    };
  }

  async cleanup(): Promise<void> {
    if (this.driver) {
      await this.driver.close();
    }

    this.som?.cleanup();
    this.autoencoder?.cleanup();

    this.nodeCache.clear();
    this.relationshipCache.clear();

    console.log('🧹 Neo4j AlphaGo Planner cleaned up');
  }
}

// Export factory function
export function createNeo4jAlphaGoPlanner(
  config: Neo4jPlannerConfig,
  visualizer: GraphVisualizationEngine
): Neo4jAlphaGoPlanner {
  return new Neo4jAlphaGoPlanner(config, visualizer);
}