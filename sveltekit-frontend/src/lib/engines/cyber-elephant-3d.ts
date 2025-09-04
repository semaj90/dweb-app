/**
 * Cyber Elephant 3D Document Visualization Engine
 * Integrates with existing Neural Sprite Engine and Tensor Upscaler
 * Provides high-dimensional document visualization with BVH spatial search
 */

import * as THREE from 'three';
import { writable, derived } from 'svelte/store';
import { tensorCoreUpscaler } from '$lib/services/tensor-upscaler-service';
import type { NeuralSpriteEngine } from './neural-sprite-engine';

export interface DocumentPoint {
  id: string;
  position: THREE.Vector3;
  embedding: Float32Array; // High-dimensional vector (1536D)
  visPosition: THREE.Vector3; // 3D projected position for visualization
  clusterId: number;
  metadata: {
    title: string;
    type: 'contract' | 'case_law' | 'evidence' | 'statute' | 'memo';
    confidence: number;
    relevance: number;
    dateCreated: string;
    size: number;
    tags: string[];
  };
}

export interface BVHNode {
  id: string;
  bounds: THREE.Box3;
  center: THREE.Vector3;
  documents: DocumentPoint[];
  children: BVHNode[];
  isLeaf: boolean;
  depth: number;
}

export interface SearchResult {
  document: DocumentPoint;
  distance: number;
  similarity: number;
  relevanceRank: number;
}

export interface CyberElephantConfig {
  maxDocuments: number;
  bvhMaxDepth: number;
  bvhLeafSize: number;
  dimensionReduction: 'umap' | 'tsne' | 'pca';
  clusteringAlgorithm: 'kmeans' | 'dbscan' | 'hierarchical';
  visualizationMode: '3d_scatter' | 'neural_network' | 'galaxy' | 'legal_timeline';
  interactionMode: 'vr_ready' | 'mouse_orbit' | 'touch_optimized';
}

export class CyberElephant3DEngine {
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private renderer: THREE.WebGLRenderer;
  private raycaster: THREE.Raycaster;
  private mouse: THREE.Vector2;
  
  // Document management
  private documents: Map<string, DocumentPoint> = new Map();
  private bvhRoot: BVHNode | null = null;
  private documentMeshes: Map<string, THREE.Mesh> = new Map();
  private clusterColors: string[] = [
    '#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57',
    '#ff9ff3', '#54a0ff', '#5f27cd', '#00d2d3', '#ff9f43'
  ];
  
  // Integration with existing systems
  private neuralEngine: NeuralSpriteEngine | null = null;
  private tensorUpscaler = tensorCoreUpscaler;
  
  // Configuration
  private config: CyberElephantConfig = {
    maxDocuments: 10000,
    bvhMaxDepth: 12,
    bvhLeafSize: 10,
    dimensionReduction: 'umap',
    clusteringAlgorithm: 'kmeans',
    visualizationMode: '3d_scatter',
    interactionMode: 'mouse_orbit'
  };
  
  // Stores for reactive updates
  public selectedDocuments = writable<DocumentPoint[]>([]);
  public searchResults = writable<SearchResult[]>([]);
  public clusterAnalysis = writable<{
    totalClusters: number;
    averageClusterSize: number;
    silhouetteScore: number;
    documentDistribution: number[];
  }>({
    totalClusters: 0,
    averageClusterSize: 0,
    silhouetteScore: 0,
    documentDistribution: []
  });
  
  public spatialMetrics = writable<{
    bvhDepth: number;
    bvhNodes: number;
    searchPerformance: number; // searches per second
    memoryUsage: number; // MB
    renderFPS: number;
  }>({
    bvhDepth: 0,
    bvhNodes: 0,
    searchPerformance: 0,
    memoryUsage: 0,
    renderFPS: 60
  });
  
  // Performance tracking
  private frameCount = 0;
  private lastFrameTime = performance.now();
  private searchCount = 0;
  private searchTimeSum = 0;
  
  constructor(canvas: HTMLCanvasElement, config?: Partial<CyberElephantConfig>) {
    this.config = { ...this.config, ...config };
    
    this.initializeThreeJS(canvas);
    this.setupEventHandlers();
    this.startPerformanceMonitoring();
    
    console.log('🐘 Cyber Elephant 3D Engine initialized');
  }
  
  /**
   * Initialize Three.js scene with optimized settings for document visualization
   */
  private initializeThreeJS(canvas: HTMLCanvasElement): void {
    // Scene setup
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0a0a0a);
    this.scene.fog = new THREE.Fog(0x0a0a0a, 100, 1000);
    
    // Camera setup (legal document viewing optimized)
    this.camera = new THREE.PerspectiveCamera(
      60, // Wider FOV for document overview
      canvas.clientWidth / canvas.clientHeight,
      0.1,
      2000
    );
    this.camera.position.set(0, 50, 100);
    this.camera.lookAt(0, 0, 0);
    
    // Renderer with WebGL2 optimization
    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      powerPreference: 'high-performance',
      stencil: false,
      depth: true
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(canvas.clientWidth, canvas.clientHeight);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    
    // Enhanced lighting for document visualization
    this.setupLighting();
    
    // Raycaster for interaction
    this.raycaster = new THREE.Raycaster();
    this.mouse = new THREE.Vector2();
  }
  
  /**
   * Setup professional lighting for legal document visualization
   */
  private setupLighting(): void {
    // Ambient light for overall visibility
    const ambientLight = new THREE.AmbientLight(0x404040, 0.3);
    this.scene.add(ambientLight);
    
    // Key light (main document illumination)
    const keyLight = new THREE.DirectionalLight(0xffffff, 0.8);
    keyLight.position.set(50, 50, 50);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.width = 2048;
    keyLight.shadow.mapSize.height = 2048;
    keyLight.shadow.camera.near = 0.5;
    keyLight.shadow.camera.far = 500;
    this.scene.add(keyLight);
    
    // Fill light (reduce harsh shadows)
    const fillLight = new THREE.DirectionalLight(0x44aaff, 0.4);
    fillLight.position.set(-50, 25, 25);
    this.scene.add(fillLight);
    
    // Rim light (document edge definition)
    const rimLight = new THREE.DirectionalLight(0xff8844, 0.3);
    rimLight.position.set(0, -50, -50);
    this.scene.add(rimLight);
    
    // Point light for cluster highlighting
    const clusterLight = new THREE.PointLight(0x00ff88, 0.5, 100);
    clusterLight.position.set(0, 25, 0);
    this.scene.add(clusterLight);
  }
  
  /**
   * Setup mouse/touch interaction handlers
   */
  private setupEventHandlers(): void {
    const canvas = this.renderer.domElement;
    
    canvas.addEventListener('mousemove', (event) => {
      const rect = canvas.getBoundingClientRect();
      this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      
      this.handleMouseMove();
    });
    
    canvas.addEventListener('click', (event) => {
      this.handleClick();
    });
    
    canvas.addEventListener('wheel', (event) => {
      this.handleZoom(event.deltaY);
    });
    
    window.addEventListener('resize', () => {
      this.handleResize();
    });
  }
  
  /**
   * Handle mouse movement for document highlighting
   */
  private handleMouseMove(): void {
    this.raycaster.setFromCamera(this.mouse, this.camera);
    const intersects = this.raycaster.intersectObjects(
      Array.from(this.documentMeshes.values())
    );
    
    // Reset all document materials
    for (const mesh of this.documentMeshes.values()) {
      const material = mesh.material as THREE.MeshStandardMaterial;
      material.emissive.setHex(0x000000);
    }
    
    // Highlight hovered document
    if (intersects.length > 0) {
      const mesh = intersects[0].object as THREE.Mesh;
      const material = mesh.material as THREE.MeshStandardMaterial;
      material.emissive.setHex(0x444444);
      
      canvas.style.cursor = 'pointer';
    } else {
      canvas.style.cursor = 'default';
    }
  }
  
  /**
   * Handle click for document selection
   */
  private handleClick(): void {
    this.raycaster.setFromCamera(this.mouse, this.camera);
    const intersects = this.raycaster.intersectObjects(
      Array.from(this.documentMeshes.values())
    );
    
    if (intersects.length > 0) {
      const mesh = intersects[0].object as THREE.Mesh;
      const documentId = mesh.userData.documentId;
      const document = this.documents.get(documentId);
      
      if (document) {
        this.selectDocument(document);
        
        // Find similar documents using BVH search
        this.findSimilarDocuments(document, 10);
        
        // Log interaction for neural engine learning
        if (this.neuralEngine) {
          this.neuralEngine.logUserActivity('document_click', {
            documentId,
            documentType: document.metadata.type,
            clusterId: document.clusterId,
            position: document.visPosition.toArray()
          });
        }
      }
    }
  }
  
  /**
   * Handle zoom interaction
   */
  private handleZoom(delta: number): void {
    const zoomSpeed = 0.1;
    const direction = this.camera.position.clone().sub(new THREE.Vector3(0, 0, 0)).normalize();
    
    if (delta > 0) {
      // Zoom out
      this.camera.position.add(direction.multiplyScalar(zoomSpeed * 10));
    } else {
      // Zoom in
      this.camera.position.sub(direction.multiplyScalar(zoomSpeed * 10));
    }
    
    // Clamp camera distance
    const distance = this.camera.position.distanceTo(new THREE.Vector3(0, 0, 0));
    if (distance < 10) {
      this.camera.position.normalize().multiplyScalar(10);
    } else if (distance > 500) {
      this.camera.position.normalize().multiplyScalar(500);
    }
  }
  
  /**
   * Handle window resize
   */
  private handleResize(): void {
    const canvas = this.renderer.domElement;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height, false);
  }
  
  /**
   * Connect to existing Neural Sprite Engine for enhanced learning
   */
  public connectNeuralEngine(engine: NeuralSpriteEngine): void {
    this.neuralEngine = engine;
    console.log('🔗 Connected to Neural Sprite Engine');
  }
  
  /**
   * Load documents and build 3D visualization
   */
  public async loadDocuments(documents: DocumentPoint[]): Promise<void> {
    console.log(`📚 Loading ${documents.length} documents into Cyber Elephant...`);
    
    // Clear existing documents
    this.clearDocuments();
    
    // Store documents
    for (const doc of documents) {
      this.documents.set(doc.id, doc);
    }
    
    // Build BVH for spatial search
    this.buildBVH();
    
    // Create 3D visualization
    this.create3DVisualization();
    
    // Update cluster analysis
    this.updateClusterAnalysis();
    
    console.log(`✅ Loaded ${this.documents.size} documents with BVH optimization`);
  }
  
  /**
   * Build BVH (Bounding Volume Hierarchy) for efficient spatial queries
   */
  private buildBVH(): void {
    const startTime = performance.now();
    const documents = Array.from(this.documents.values());
    
    if (documents.length === 0) return;
    
    // Calculate bounding box for all documents
    const bounds = new THREE.Box3();
    for (const doc of documents) {
      bounds.expandByPoint(doc.visPosition);
    }
    
    // Build BVH recursively
    this.bvhRoot = this.buildBVHNode(documents, bounds, 0);
    
    const buildTime = performance.now() - startTime;
    console.log(`🌳 BVH built in ${buildTime.toFixed(2)}ms with ${this.countBVHNodes(this.bvhRoot)} nodes`);
    
    // Update spatial metrics
    this.spatialMetrics.update(metrics => ({
      ...metrics,
      bvhDepth: this.calculateBVHDepth(this.bvhRoot),
      bvhNodes: this.countBVHNodes(this.bvhRoot)
    }));
  }
  
  /**
   * Build individual BVH node recursively
   */
  private buildBVHNode(documents: DocumentPoint[], bounds: THREE.Box3, depth: number): BVHNode {
    const node: BVHNode = {
      id: `bvh_${depth}_${Math.random().toString(36).substr(2, 9)}`,
      bounds: bounds.clone(),
      center: bounds.getCenter(new THREE.Vector3()),
      documents: [],
      children: [],
      isLeaf: false,
      depth
    };
    
    // Stopping criteria
    if (documents.length <= this.config.bvhLeafSize || depth >= this.config.bvhMaxDepth) {
      node.documents = documents;
      node.isLeaf = true;
      return node;
    }
    
    // Find split axis (longest dimension)
    const size = bounds.getSize(new THREE.Vector3());
    let splitAxis = 0; // x
    if (size.y > size.x && size.y > size.z) splitAxis = 1; // y
    else if (size.z > size.x && size.z > size.y) splitAxis = 2; // z
    
    // Sort documents by split axis
    documents.sort((a, b) => {
      const posA = splitAxis === 0 ? a.visPosition.x : splitAxis === 1 ? a.visPosition.y : a.visPosition.z;
      const posB = splitAxis === 0 ? b.visPosition.x : splitAxis === 1 ? b.visPosition.y : b.visPosition.z;
      return posA - posB;
    });
    
    // Split documents
    const mid = Math.floor(documents.length / 2);
    const leftDocs = documents.slice(0, mid);
    const rightDocs = documents.slice(mid);
    
    // Calculate child bounds
    const leftBounds = new THREE.Box3();
    const rightBounds = new THREE.Box3();
    
    for (const doc of leftDocs) leftBounds.expandByPoint(doc.visPosition);
    for (const doc of rightDocs) rightBounds.expandByPoint(doc.visPosition);
    
    // Create child nodes
    if (leftDocs.length > 0) {
      node.children.push(this.buildBVHNode(leftDocs, leftBounds, depth + 1));
    }
    if (rightDocs.length > 0) {
      node.children.push(this.buildBVHNode(rightDocs, rightBounds, depth + 1));
    }
    
    return node;
  }
  
  /**
   * Count total BVH nodes
   */
  private countBVHNodes(node: BVHNode): number {
    if (!node) return 0;
    return 1 + node.children.reduce((sum, child) => sum + this.countBVHNodes(child), 0);
  }
  
  /**
   * Calculate BVH depth
   */
  private calculateBVHDepth(node: BVHNode): number {
    if (!node) return 0;
    if (node.isLeaf) return node.depth;
    return Math.max(...node.children.map(child => this.calculateBVHDepth(child)));
  }
  
  /**
   * Create 3D visualization of documents
   */
  private create3DVisualization(): void {
    const geometry = new THREE.SphereGeometry(0.8, 16, 16);
    
    for (const [docId, doc] of this.documents) {
      // Create document material based on type and cluster
      const material = new THREE.MeshStandardMaterial({
        color: this.getDocumentColor(doc),
        metalness: 0.1,
        roughness: 0.4,
        transparent: true,
        opacity: 0.8
      });
      
      // Create document mesh
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.copy(doc.visPosition);
      mesh.scale.setScalar(this.getDocumentScale(doc));
      mesh.userData = {
        documentId: docId,
        type: doc.metadata.type,
        clusterId: doc.clusterId
      };
      
      // Add to scene and store reference
      this.scene.add(mesh);
      this.documentMeshes.set(docId, mesh);
    }
    
    // Add cluster connections if in neural network mode
    if (this.config.visualizationMode === 'neural_network') {
      this.addClusterConnections();
    }
    
    console.log(`🎨 Created 3D visualization with ${this.documentMeshes.size} document meshes`);
  }
  
  /**
   * Get document color based on type and cluster
   */
  private getDocumentColor(doc: DocumentPoint): string {
    // Primary color from cluster
    const clusterColor = this.clusterColors[doc.clusterId % this.clusterColors.length];
    
    // Modify based on document type
    const typeModifiers = {
      contract: 1.0,      // Pure cluster color
      case_law: 0.8,      // Slightly darker
      evidence: 1.2,      // Slightly brighter
      statute: 0.9,       // Darker
      memo: 1.1           // Brighter
    };
    
    const modifier = typeModifiers[doc.metadata.type] || 1.0;
    return clusterColor; // Color modification would be done with THREE.Color
  }
  
  /**
   * Get document scale based on relevance and confidence
   */
  private getDocumentScale(doc: DocumentPoint): number {
    const baseScale = 1.0;
    const relevanceScale = 0.5 + (doc.metadata.relevance * 0.5); // 0.5x to 1x
    const confidenceScale = 0.8 + (doc.metadata.confidence * 0.4); // 0.8x to 1.2x
    
    return baseScale * relevanceScale * confidenceScale;
  }
  
  /**
   * Add cluster connections for neural network visualization
   */
  private addClusterConnections(): void {
    // Group documents by cluster
    const clusters = new Map<number, DocumentPoint[]>();
    for (const doc of this.documents.values()) {
      if (!clusters.has(doc.clusterId)) {
        clusters.set(doc.clusterId, []);
      }
      clusters.get(doc.clusterId)!.push(doc);
    }
    
    // Create connections within clusters
    const lineMaterial = new THREE.LineBasicMaterial({
      color: 0x444444,
      transparent: true,
      opacity: 0.3
    });
    
    for (const [clusterId, clusterDocs] of clusters) {
      if (clusterDocs.length < 2) continue;
      
      // Calculate cluster center
      const center = new THREE.Vector3();
      for (const doc of clusterDocs) {
        center.add(doc.visPosition);
      }
      center.divideScalar(clusterDocs.length);
      
      // Connect each document to cluster center
      for (const doc of clusterDocs) {
        const geometry = new THREE.BufferGeometry().setFromPoints([
          doc.visPosition,
          center
        ]);
        const line = new THREE.Line(geometry, lineMaterial);
        this.scene.add(line);
      }
    }
  }
  
  /**
   * Perform BVH-accelerated nearest neighbor search
   */
  public findNearestDocuments(queryPoint: THREE.Vector3, k: number = 10): SearchResult[] {
    const startTime = performance.now();
    
    if (!this.bvhRoot) return [];
    
    const candidates: { document: DocumentPoint; distance: number }[] = [];
    this.searchBVHNode(this.bvhRoot, queryPoint, candidates);
    
    // Sort by distance and take top k
    candidates.sort((a, b) => a.distance - b.distance);
    const nearestCandidates = candidates.slice(0, k);
    
    // Convert to SearchResult format
    const results: SearchResult[] = nearestCandidates.map((candidate, index) => ({
      document: candidate.document,
      distance: candidate.distance,
      similarity: 1 / (1 + candidate.distance), // Convert distance to similarity
      relevanceRank: index + 1
    }));
    
    const searchTime = performance.now() - startTime;
    this.searchCount++;
    this.searchTimeSum += searchTime;
    
    console.log(`🔍 BVH search found ${results.length} documents in ${searchTime.toFixed(2)}ms`);
    
    return results;
  }
  
  /**
   * Search BVH node recursively
   */
  private searchBVHNode(
    node: BVHNode, 
    queryPoint: THREE.Vector3, 
    candidates: { document: DocumentPoint; distance: number }[]
  ): void {
    // Calculate distance to bounding box
    const distanceToBounds = node.bounds.distanceToPoint(queryPoint);
    
    // Early termination if this subtree is too far
    if (candidates.length >= 100 && distanceToBounds > candidates[candidates.length - 1].distance) {
      return;
    }
    
    if (node.isLeaf) {
      // Add all documents in leaf node
      for (const doc of node.documents) {
        const distance = queryPoint.distanceTo(doc.visPosition);
        candidates.push({ document: doc, distance });
      }
    } else {
      // Search child nodes (closer first)
      const childDistances = node.children.map(child => ({
        child,
        distance: child.bounds.distanceToPoint(queryPoint)
      }));
      
      childDistances.sort((a, b) => a.distance - b.distance);
      
      for (const { child } of childDistances) {
        this.searchBVHNode(child, queryPoint, candidates);
      }
    }
  }
  
  /**
   * Find documents similar to a given document
   */
  public findSimilarDocuments(queryDoc: DocumentPoint, k: number = 10): void {
    const results = this.findNearestDocuments(queryDoc.visPosition, k + 1); // +1 to exclude self
    const filteredResults = results.filter(result => result.document.id !== queryDoc.id);
    
    this.searchResults.set(filteredResults.slice(0, k));
    this.highlightSearchResults(filteredResults);
  }
  
  /**
   * Highlight search results in 3D visualization
   */
  private highlightSearchResults(results: SearchResult[]): void {
    // Reset all highlights
    for (const mesh of this.documentMeshes.values()) {
      const material = mesh.material as THREE.MeshStandardMaterial;
      material.emissive.setHex(0x000000);
      mesh.scale.setScalar(1.0);
    }
    
    // Highlight results with decreasing intensity
    results.forEach((result, index) => {
      const mesh = this.documentMeshes.get(result.document.id);
      if (mesh) {
        const material = mesh.material as THREE.MeshStandardMaterial;
        const intensity = Math.max(0.1, 1 - (index / results.length));
        material.emissive.setHex(0x00ff88);
        material.emissiveIntensity = intensity;
        mesh.scale.setScalar(1 + intensity * 0.5);
      }
    });
  }
  
  /**
   * Select a document and update UI
   */
  private selectDocument(document: DocumentPoint): void {
    this.selectedDocuments.update(docs => {
      const isSelected = docs.some(doc => doc.id === document.id);
      if (isSelected) {
        return docs.filter(doc => doc.id !== document.id); // Deselect
      } else {
        return [...docs, document]; // Select
      }
    });
  }
  
  /**
   * Update cluster analysis metrics
   */
  private updateClusterAnalysis(): void {
    const clusterMap = new Map<number, number>();
    
    for (const doc of this.documents.values()) {
      clusterMap.set(doc.clusterId, (clusterMap.get(doc.clusterId) || 0) + 1);
    }
    
    const clusters = Array.from(clusterMap.values());
    const totalClusters = clusters.length;
    const averageClusterSize = clusters.reduce((sum, size) => sum + size, 0) / totalClusters;
    
    // Calculate silhouette score (simplified version)
    const silhouetteScore = this.calculateSilhouetteScore();
    
    this.clusterAnalysis.set({
      totalClusters,
      averageClusterSize,
      silhouetteScore,
      documentDistribution: clusters
    });
  }
  
  /**
   * Calculate simplified silhouette score for cluster quality
   */
  private calculateSilhouetteScore(): number {
    // Simplified calculation - in practice would use proper silhouette analysis
    const documents = Array.from(this.documents.values());
    let totalScore = 0;
    
    for (const doc of documents) {
      // Calculate average intra-cluster distance
      const sameClusterDocs = documents.filter(d => d.clusterId === doc.clusterId && d.id !== doc.id);
      const intraClusterDistance = sameClusterDocs.length > 0 
        ? sameClusterDocs.reduce((sum, d) => sum + doc.visPosition.distanceTo(d.visPosition), 0) / sameClusterDocs.length
        : 0;
      
      // Calculate minimum average inter-cluster distance
      const otherClusters = new Set(documents.map(d => d.clusterId).filter(id => id !== doc.clusterId));
      let minInterClusterDistance = Infinity;
      
      for (const clusterId of otherClusters) {
        const otherClusterDocs = documents.filter(d => d.clusterId === clusterId);
        const avgDistance = otherClusterDocs.reduce((sum, d) => sum + doc.visPosition.distanceTo(d.visPosition), 0) / otherClusterDocs.length;
        minInterClusterDistance = Math.min(minInterClusterDistance, avgDistance);
      }
      
      // Calculate silhouette for this document
      const silhouette = (minInterClusterDistance - intraClusterDistance) / Math.max(intraClusterDistance, minInterClusterDistance);
      totalScore += silhouette;
    }
    
    return totalScore / documents.length;
  }
  
  /**
   * Clear all documents and visualization
   */
  private clearDocuments(): void {
    // Remove meshes from scene
    for (const mesh of this.documentMeshes.values()) {
      this.scene.remove(mesh);
      mesh.geometry.dispose();
      if (Array.isArray(mesh.material)) {
        mesh.material.forEach(material => material.dispose());
      } else {
        mesh.material.dispose();
      }
    }
    
    this.documents.clear();
    this.documentMeshes.clear();
    this.bvhRoot = null;
    
    this.selectedDocuments.set([]);
    this.searchResults.set([]);
  }
  
  /**
   * Start performance monitoring loop
   */
  private startPerformanceMonitoring(): void {
    setInterval(() => {
      // Calculate average search performance
      const avgSearchPerformance = this.searchCount > 0 
        ? 1000 / (this.searchTimeSum / this.searchCount)
        : 0;
      
      // Estimate memory usage (rough calculation)
      const memoryUsage = (
        this.documents.size * 0.05 + // ~50KB per document
        (this.bvhRoot ? this.countBVHNodes(this.bvhRoot) * 0.001 : 0) // ~1KB per BVH node
      );
      
      this.spatialMetrics.update(metrics => ({
        ...metrics,
        searchPerformance: avgSearchPerformance,
        memoryUsage,
        renderFPS: this.calculateFPS()
      }));
      
      // Reset counters
      this.searchCount = 0;
      this.searchTimeSum = 0;
    }, 1000);
  }
  
  /**
   * Calculate current FPS
   */
  private calculateFPS(): number {
    const now = performance.now();
    const deltaTime = now - this.lastFrameTime;
    this.lastFrameTime = now;
    this.frameCount++;
    
    return Math.round(1000 / deltaTime);
  }
  
  /**
   * Render the scene
   */
  public render(): void {
    // Integration with Tensor Upscaler for enhanced rendering
    if (this.tensorUpscaler && this.config.interactionMode === 'vr_ready') {
      // Apply tensor upscaling for VR-quality rendering
      this.tensorUpscaler.optimizeForMetricsDashboard();
    }
    
    this.renderer.render(this.scene, this.camera);
  }
  
  /**
   * Animation loop
   */
  public animate(): void {
    requestAnimationFrame(() => this.animate());
    
    // Auto-rotate camera if no interaction
    if (this.config.visualizationMode === 'galaxy') {
      this.camera.position.applyAxisAngle(new THREE.Vector3(0, 1, 0), 0.005);
      this.camera.lookAt(0, 0, 0);
    }
    
    this.render();
  }
  
  /**
   * Update configuration
   */
  public updateConfig(newConfig: Partial<CyberElephantConfig>): void {
    this.config = { ...this.config, ...newConfig };
    
    // Rebuild visualization if needed
    if (this.documents.size > 0) {
      this.create3DVisualization();
    }
  }
  
  /**
   * Get system status
   */
  public getStatus(): {
    documentsLoaded: number;
    bvhBuilt: boolean;
    visualizationMode: string;
    performanceGrade: 'A' | 'B' | 'C' | 'D' | 'F';
  } {
    let grade: 'A' | 'B' | 'C' | 'D' | 'F' = 'F';
    const metrics = this.spatialMetrics;
    
    // Grade based on performance (this would access current values)
    const fps = 60; // Would get from current metrics
    const searchPerf = 100; // Would get from current metrics
    
    if (fps >= 55 && searchPerf >= 80) grade = 'A';
    else if (fps >= 45 && searchPerf >= 60) grade = 'B';
    else if (fps >= 30 && searchPerf >= 40) grade = 'C';
    else if (fps >= 20 && searchPerf >= 20) grade = 'D';
    
    return {
      documentsLoaded: this.documents.size,
      bvhBuilt: this.bvhRoot !== null,
      visualizationMode: this.config.visualizationMode,
      performanceGrade: grade
    };
  }
  
  /**
   * Cleanup resources
   */
  public destroy(): void {
    console.log('🛑 Shutting down Cyber Elephant 3D Engine...');
    
    this.clearDocuments();
    
    // Dispose of Three.js resources
    this.renderer.dispose();
    
    console.log('✅ Cyber Elephant 3D Engine destroyed');
  }
}

// Factory function for easy integration
export function createCyberElephant3D(
  canvas: HTMLCanvasElement, 
  config?: Partial<CyberElephantConfig>
): CyberElephant3DEngine {
  return new CyberElephant3DEngine(canvas, config);
}

// Mock data generator for testing
export function generateMockDocuments(count: number): DocumentPoint[] {
  const types: Array<'contract' | 'case_law' | 'evidence' | 'statute' | 'memo'> = 
    ['contract', 'case_law', 'evidence', 'statute', 'memo'];
  
  const documents: DocumentPoint[] = [];
  
  for (let i = 0; i < count; i++) {
    // Generate high-dimensional embedding (mock)
    const embedding = new Float32Array(1536);
    for (let j = 0; j < 1536; j++) {
      embedding[j] = Math.random() * 2 - 1; // -1 to 1
    }
    
    // Project to 3D using PCA-like transformation (mock)
    const visPosition = new THREE.Vector3(
      Math.random() * 200 - 100, // -100 to 100
      Math.random() * 200 - 100,
      Math.random() * 200 - 100
    );
    
    // Cluster using simple spatial grouping
    const clusterId = Math.floor(visPosition.length() / 20) % 8;
    
    documents.push({
      id: `doc_${i}`,
      position: visPosition.clone(),
      embedding,
      visPosition,
      clusterId,
      metadata: {
        title: `Document ${i + 1}`,
        type: types[Math.floor(Math.random() * types.length)],
        confidence: 0.7 + Math.random() * 0.3,
        relevance: Math.random(),
        dateCreated: new Date(2020 + Math.random() * 4, 0, 1).toISOString(),
        size: 1000 + Math.floor(Math.random() * 10000),
        tags: [`tag_${Math.floor(Math.random() * 10)}`, `category_${Math.floor(Math.random() * 5)}`]
      }
    });
  }
  
  console.log(`🎲 Generated ${count} mock documents for testing`);
  return documents;
}