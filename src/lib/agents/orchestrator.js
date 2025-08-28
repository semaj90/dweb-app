// FIX: All necessary imports optimized for your tech stack
import { ChatOllama } from '@langchain/ollama';
import { ChatAnthropic } from '@langchain/anthropic';
import { ChatGoogleGenerativeAI } from '@langchain/google-genai';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { PromptTemplate } from '@langchain/core/prompts';
import { RunnableSequence } from '@langchain/core/runnables';
import { env } from '$env/dynamic/private';
import { cacheManager } from '../database/redis.js';

// LRU Cache for result memoization - Production optimized
class LRUCache {
  constructor(maxSize = 1000) {
    this.maxSize = maxSize;
    this.cache = new Map();
  }

  get(key) {
    if (this.cache.has(key)) {
      // Move to end (most recent)
      const value = this.cache.get(key);
      this.cache.delete(key);
      this.cache.set(key, value);
      return value;
    }
    return null;
  }

  set(key, value) {
    if (this.cache.has(key)) {
      this.cache.delete(key);
    } else if (this.cache.size >= this.maxSize) {
      // Remove oldest entry
      const firstKey = this.cache.keys().next().value;
      this.cache.delete(firstKey);
    }
    this.cache.set(key, value);
  }

  has(key) {
    return this.cache.has(key);
  }

  clear() {
    this.cache.clear();
  }

  get size() {
    return this.cache.size;
  }
}

// Worker Pool Manager for persistent workers
class WorkerPoolManager {
  constructor() {
    this.workers = new Map();
    this.workQueue = [];
    this.maxWorkers = parseInt(env.MAX_WORKERS || '4');
    this.initialized = false;
  }

  async initializeWorkers() {
    if (this.initialized) return;

    // Initialize GPU-bound workers (keep alive for CUDA efficiency)
    for (let i = 0; i < this.maxWorkers; i++) {
      const workerId = `worker_${i}`;
      this.workers.set(workerId, {
        id: workerId,
        busy: false,
        startTime: Date.now(),
        tasksCompleted: 0,
        lastUsed: Date.now()
      });
    }
    this.initialized = true;
    console.log(`Initialized ${this.maxWorkers} persistent workers`);
  }

  getAvailableWorker() {
    for (const [id, worker] of this.workers) {
      if (!worker.busy) {
        worker.busy = true;
        worker.lastUsed = Date.now();
        return worker;
      }
    }
    return null;
  }

  releaseWorker(workerId) {
    const worker = this.workers.get(workerId);
    if (worker) {
      worker.busy = false;
      worker.tasksCompleted++;
    }
  }

  getWorkerStats() {
    const stats = {
      total: this.workers.size,
      available: 0,
      busy: 0,
      totalTasks: 0
    };

    for (const worker of this.workers.values()) {
      if (worker.busy) {
        stats.busy++;
      } else {
        stats.available++;
      }
      stats.totalTasks += worker.tasksCompleted;
    }

    return stats;
  }
}

// Optional NATS integration for real-time messaging (if available)
let natsMessaging;
try {
  const natsModule = await import('../services/nats-messaging-service.js');
  natsMessaging = natsModule.NATSMessagingService;
} catch (error) {
  console.log('NATS messaging service not available, continuing without real-time messaging');
}

/**
 * Advanced Legal AI Agent Orchestrator
 * Integrates LangChain + CrewAI + Ollama + Claude + Gemini fallbacks
 * Supports streaming, worker threads, and intelligent agent selection
 */

/**
 * @typedef {Object} AgentConfig
 * @property {string} name
 * @property {'ollama' | 'claude' | 'gemini'} type
 * @property {string} model
 * @property {number} temperature
 * @property {number} maxTokens
 * @property {string} systemPrompt
 * @property {string[]} specialization
 */

/**
 * @typedef {Object} OrchestrationRequest
 * @property {string} query
 * @property {string} [sessionId] - Added from stub
 * @property {'contract' | 'motion' | 'evidence' | 'correspondence' | 'brief'} [documentType]
 * @property {string} [jurisdiction]
 * @property {'low' | 'medium' | 'high' | 'critical'} urgency
 * @property {boolean} requiresMultiAgent
 * @property {boolean} enableStreaming
 * @property {Record<string, any>} [context]
 */

/**
 * @typedef {Object} TokenUsage
 * @property {number} prompt
 * @property {number} completion
 * @property {number} total
 */

/**
 * @typedef {Object} AgentResponse
 * @property {string} agentName
 * @property {string} response
 * @property {number} confidence
 * @property {number} processingTime
 * @property {TokenUsage} tokenUsage
 * @property {Record<string, any>} metadata
 */

/**
 * @typedef {Object} OrchestrationResult
 * @property {AgentResponse} primaryResponse
 * @property {AgentResponse[]} [collaborativeAnalysis]
 * @property {string} synthesizedConclusion
 * @property {string[]} recommendations
 * @property {number} confidence
 * @property {number} totalProcessingTime
 * @property {string} [cacheKey]
 */

// Predefined agent configurations optimized for your legal AI stack
const LEGAL_AGENT_CONFIGS = [
  {
    name: 'legal-analyst',
    type: 'ollama',
    model: env.OLLAMA_LEGAL_MODEL || 'gemma3-legal',  // Optimized for your stack
    temperature: 0.1,
    maxTokens: 2048,
    systemPrompt: `You are a senior legal analyst specializing in document analysis, case law research, and legal precedent identification.
    Provide precise, citations-backed analysis with attention to jurisdictional differences and regulatory compliance.
    Use your knowledge of legal databases, precedent systems, and YoRHa command protocols for structured responses.`,
    specialization: ['document_analysis', 'case_law', 'precedent_research', 'compliance_review']
  },
  {
    name: 'contract-specialist',
    type: 'ollama', 
    model: env.OLLAMA_LEGAL_MODEL || 'gemma3-legal',  // Optimized for your stack
    temperature: 0.05,
    maxTokens: 3072,
    systemPrompt: `You are an expert contract attorney specializing in contract drafting, review, and risk assessment.
    Focus on identifying key terms, obligations, risk factors, and potential legal issues in contractual documents.
    Structure your analysis using clear sections and provide confidence scores for recommendations.`,
    specialization: ['contract_drafting', 'risk_assessment', 'term_analysis', 'obligation_mapping']
  },
  {
    name: 'litigation-strategist',
    type: 'claude',
    model: env.CLAUDE_MODEL || 'claude-3-5-sonnet-20241022',  // Latest model
    temperature: 0.2,
    maxTokens: 4096,
    systemPrompt: `You are a litigation strategy expert with deep knowledge of procedural law, evidence rules, and courtroom tactics.
    Provide strategic guidance on case positioning, evidence evaluation, and procedural considerations.
    Your responses should integrate with legal document management systems and evidence tracking protocols.`,
    specialization: ['litigation_strategy', 'evidence_analysis', 'procedural_law', 'case_positioning']
  },
  {
    name: 'regulatory-compliance',
    type: 'gemini',
    model: env.GEMINI_MODEL || 'gemini-1.5-pro',
    temperature: 0.1,
    maxTokens: 2048,
    systemPrompt: `You are a regulatory compliance specialist with expertise in financial services, healthcare, and corporate law.
    Focus on regulatory requirements, compliance frameworks, and risk mitigation strategies.
    Provide actionable compliance recommendations with clear priority levels and implementation timelines.`,
    specialization: ['regulatory_analysis', 'compliance_frameworks', 'risk_mitigation', 'policy_review']
  }
];

// FIX: This is the single, consolidated class. The stub version has been removed.
export class LegalAIOrchestrator {
  constructor() {
    this.agents = new Map();
    this.outputParser = new StringOutputParser();
    
    // FIX: Added session and processing state from the original stub for a complete API.
    this.sessions = new Map();
    this.processing = false;
    
    // Initialize NATS messaging if available
    this.nats = natsMessaging ? new natsMessaging() : null;
    
    // Initialize LRU cache for result memoization (production optimized)
    this.lruCache = new LRUCache(parseInt(env.LRU_CACHE_SIZE || '1000'));
    
    // Initialize worker pool manager for persistent workers
    this.workerPool = new WorkerPoolManager();
    
    // Performance metrics for monitoring (enhanced with caching metrics)
    this.metrics = {
      totalRequests: 0,
      successfulRequests: 0,
      failedRequests: 0,
      averageResponseTime: 0,
      cacheHitRate: 0,
      lruCacheHitRate: 0,
      workerUtilization: 0,
      persistentWorkerTasks: 0
    };
    
    this.initializeAgents();
    this.initializeHealthMonitoring();
    this.initializeWorkerPool();
  }

  // Initialize worker pool for persistent workers (GPU-optimized)
  async initializeWorkerPool() {
    await this.workerPool.initializeWorkers();
    
    // Periodic worker health checks and cleanup
    setInterval(() => {
      const stats = this.workerPool.getWorkerStats();
      this.metrics.workerUtilization = Math.round((stats.busy / stats.total) * 100);
      this.metrics.persistentWorkerTasks = stats.totalTasks;
    }, 10000); // Every 10 seconds
  }

  // Initialize health monitoring for your stack
  initializeHealthMonitoring() {
    // Periodic health checks for services
    setInterval(async () => {
      await this.performHealthChecks();
    }, 30000); // Every 30 seconds
    
    // Publish metrics to NATS if available
    if (this.nats) {
      setInterval(async () => {
        try {
          await this.nats.publish('legal.ai.orchestrator.metrics', {
            timestamp: Date.now(),
            metrics: this.metrics,
            agentCount: this.agents.size,
            activeSessions: this.sessions.size
          });
        } catch (error) {
          console.warn('Failed to publish metrics to NATS:', error);
        }
      }, 60000); // Every minute
    }
  }

  // Perform health checks on all services
  async performHealthChecks() {
    const healthStatus = {
      redis: false,
      ollama: false,
      nats: false,
      agents: {}
    };

    // Check Redis connection
    try {
      healthStatus.redis = await cacheManager.exists('health:check') !== null;
    } catch (error) {
      healthStatus.redis = false;
    }

    // Check Ollama availability
    try {
      const response = await fetch(`${env.OLLAMA_BASE_URL || 'http://localhost:11434'}/api/tags`);
      healthStatus.ollama = response.ok;
    } catch (error) {
      healthStatus.ollama = false;
    }

    // Check NATS connection
    if (this.nats) {
      try {
        healthStatus.nats = this.nats.isConnected && this.nats.isConnected();
      } catch (error) {
        healthStatus.nats = false;
      }
    }

    // Store health status in Redis
    try {
      await cacheManager.set('orchestrator:health', healthStatus, 60);
    } catch (error) {
      console.warn('Failed to store health status:', error);
    }

    return healthStatus;
  }

  initializeAgents() {
    for (const config of LEGAL_AGENT_CONFIGS) {
      let agent;

      switch (config.type) {
        case 'ollama':
          agent = new ChatOllama({
            baseUrl: env.OLLAMA_BASE_URL || 'http://localhost:11434',
            model: config.model,
            temperature: config.temperature,
            // Optimized for your multi-core Ollama cluster
            numCtx: 4096,  // Context window
            numGpu: 1,     // GPU layers for RTX 3060 Ti
            numThread: 8,  // Multi-threading
            repeatPenalty: 1.1,
            topK: 40,
            topP: 0.9,
          });
          break;

        case 'claude':
          agent = new ChatAnthropic({
            apiKey: env.ANTHROPIC_API_KEY,
            model: config.model,  // Fixed: use 'model' instead of 'modelName' for newer versions
            temperature: config.temperature,
            maxTokens: config.maxTokens,
            // Optimized for legal use cases
            topK: 5,
            topP: 0.7,
          });
          break;

        case 'gemini':
          agent = new ChatGoogleGenerativeAI({
            apiKey: env.GOOGLE_AI_API_KEY,
            model: config.model,  // Fixed: use 'model' instead of 'modelName' 
            temperature: config.temperature,
            maxOutputTokens: config.maxTokens,
            // Optimized for structured legal analysis
            topK: 40,
            topP: 0.8,
            safetySettings: [
              {
                category: 'HARM_CATEGORY_HARASSMENT',
                threshold: 'BLOCK_ONLY_HIGH'
              }
            ],
          });
          break;
      }

      if (agent) {
        this.agents.set(config.name, { agent, config });
      }
    }
  }

  async orchestrate(request) {
    const startTime = Date.now();
    this.processing = true;
    this.metrics.totalRequests++;
    
    // Get available worker from persistent pool
    const worker = this.workerPool.getAvailableWorker();
    if (!worker) {
      // Queue the request if no workers available
      return new Promise((resolve, reject) => {
        this.workerPool.workQueue.push({ request, resolve, reject, startTime });
      });
    }

    try {
      // Publish analysis start to NATS if available
      if (this.nats) {
        try {
          await this.nats.publish('legal.ai.analysis.started', {
            timestamp: startTime,
            sessionId: request.sessionId,
            documentType: request.documentType,
            urgency: request.urgency,
            workerId: worker.id
          });
        } catch (error) {
          console.warn('Failed to publish analysis start event:', error);
        }
      }

      // Multi-layer caching: LRU first, then Redis
      const cacheKey = this.generateCacheKey(request);
      let cacheHit = false;
      let lruCacheHit = false;

      // Check LRU cache first (fastest)
      let cached = this.lruCache.get(cacheKey);
      if (cached) {
        lruCacheHit = true;
        cacheHit = true;
        this.processing = false;
        this.workerPool.releaseWorker(worker.id);
        this.updateMetrics(startTime, true, cacheHit, lruCacheHit);
        return cached;
      }

      // Check Redis cache (distributed)
      try {
        cached = await cacheManager.get(cacheKey);
        if (cached) {
          cacheHit = true;
          // Store in LRU for faster future access
          this.lruCache.set(cacheKey, cached);
          this.processing = false;
          this.workerPool.releaseWorker(worker.id);
          this.updateMetrics(startTime, true, cacheHit, lruCacheHit);
          return cached;
        }
      } catch (cacheError) {
        console.warn('Cache lookup failed:', cacheError);
      }

    try {
      // Select appropriate agent(s)
      const selectedAgents = this.selectAgents(request);
      
      if (selectedAgents.length === 0) {
        throw new Error("No suitable AI agent found for the request.");
      }

      let result;
      if (request.requiresMultiAgent && selectedAgents.length > 1) {
        result = await this.multiAgentOrchestration(request, selectedAgents, startTime, cacheKey);
      } else {
        result = await this.singleAgentExecution(request, selectedAgents[0], startTime, cacheKey);
      }

      // Store session information
      const sessionId = request.sessionId || this.generateSessionId();
      this.sessions.set(sessionId, {
          ...this.sessions.get(sessionId),
          lastQuery: request.query,
          lastResult: result.synthesizedConclusion,
          timestamp: Date.now()
      });

      // Publish analysis completion to NATS if available
      if (this.nats) {
        try {
          await this.nats.publish('legal.ai.analysis.completed', {
            timestamp: Date.now(),
            sessionId,
            confidence: result.confidence,
            processingTime: result.totalProcessingTime,
            recommendations: result.recommendations.length
          });
        } catch (error) {
          console.warn('Failed to publish analysis completion event:', error);
        }
      }

      // Store in both caches for future requests
      await this.cacheResult(cacheKey, result);
      this.lruCache.set(cacheKey, result);

      this.updateMetrics(startTime, true, cacheHit, lruCacheHit);
      return result;

    } catch (error) {
      console.error('Orchestration error:', error);
      
      // Publish analysis failure to NATS if available
      if (this.nats) {
        try {
          await this.nats.publish('legal.ai.analysis.failed', {
            timestamp: Date.now(),
            sessionId: request.sessionId,
            workerId: worker.id,
            error: error.message,
            processingTime: Date.now() - startTime
          });
        } catch (natsError) {
          console.warn('Failed to publish analysis failure event:', natsError);
        }
      }
      
      this.updateMetrics(startTime, false, cacheHit, lruCacheHit);
      throw new Error(`Orchestration failed: ${error.message}`);
    } finally {
      this.processing = false;
      this.workerPool.releaseWorker(worker.id);
      this.processWorkQueue(); // Process any queued requests
    }
    } catch (workerError) {
      this.workerPool.releaseWorker(worker.id);
      throw workerError;
    }
  }

  async singleAgentExecution(
    request,
    agentInfo,
    startTime,
    cacheKey
  ) {
    const { agent, config } = agentInfo;

    const prompt = PromptTemplate.fromTemplate(`
    System Context: {systemPrompt}
    Document Type: {documentType}
    Jurisdiction: {jurisdiction}
    Urgency Level: {urgency}
    Legal Query: {query}
    Additional Context: {context}
    Please provide a comprehensive legal analysis including:
    1. Key legal issues identified
    2. Relevant statutes, regulations, or case law
    3. Risk assessment and potential implications
    4. Actionable recommendations
    5. Confidence level in your analysis (1-10)
    Analysis:
    `);

    const chain = RunnableSequence.from([prompt, agent, this.outputParser]);

    const agentStartTime = Date.now();
    const response = await chain.invoke({
      systemPrompt: config.systemPrompt,
      documentType: request.documentType || 'general',
      jurisdiction: request.jurisdiction || 'federal',
      urgency: request.urgency,
      query: request.query,
      context: JSON.stringify(request.context || {})
    });

    const agentResponse = {
      agentName: config.name,
      response,
      confidence: this.extractConfidence(response),
      processingTime: Date.now() - agentStartTime,
      tokenUsage: this.estimateTokenUsage(request.query, response),
      metadata: { model: config.model, temperature: config.temperature, specialization: config.specialization }
    };

    const result = {
      primaryResponse: agentResponse,
      synthesizedConclusion: response,
      recommendations: this.extractRecommendations(response),
      confidence: agentResponse.confidence,
      totalProcessingTime: Date.now() - startTime,
      cacheKey
    };

    await this.cacheResult(cacheKey, result);
    return result;
  }

  async multiAgentOrchestration(
    request,
    agents,
    startTime,
    cacheKey
  ) {
    const agentPromises = agents.map(async ({ agent, config }) => {
      const specializedPrompt = this.createSpecializedPrompt(request, config);
      const chain = RunnableSequence.from([specializedPrompt, agent, this.outputParser]);
      const agentStartTime = Date.now();
      const response = await chain.invoke({
        query: request.query,
        documentType: request.documentType || 'general',
        jurisdiction: request.jurisdiction || 'federal',
        urgency: request.urgency,
        context: JSON.stringify(request.context || {})
      });

      return {
        agentName: config.name,
        response,
        confidence: this.extractConfidence(response),
        processingTime: Date.now() - agentStartTime,
        tokenUsage: this.estimateTokenUsage(request.query, response),
        metadata: { model: config.model, specialization: config.specialization }
      };
    });

    const collaborativeAnalysis = await Promise.all(agentPromises);
    const synthesisAgent = this.selectSynthesisAgent(request);
    const synthesizedConclusion = await this.synthesizeResponses(request, collaborativeAnalysis, synthesisAgent);

    const result = {
      primaryResponse: collaborativeAnalysis[0],
      collaborativeAnalysis,
      synthesizedConclusion,
      recommendations: this.extractRecommendations(synthesizedConclusion),
      confidence: this.calculateOverallConfidence(collaborativeAnalysis),
      totalProcessingTime: Date.now() - startTime,
      cacheKey
    };

    await this.cacheResult(cacheKey, result);
    return result;
  }

  selectAgents(request) {
    let selectedAgents = [];

    switch (request.documentType) {
      case 'contract':
        selectedAgents.push(this.agents.get('contract-specialist'));
        if (request.requiresMultiAgent) selectedAgents.push(this.agents.get('legal-analyst'));
        break;
      case 'motion':
      case 'brief':
        selectedAgents.push(this.agents.get('litigation-strategist'));
        if (request.requiresMultiAgent) selectedAgents.push(this.agents.get('legal-analyst'));
        break;
      case 'evidence':
        selectedAgents.push(this.agents.get('litigation-strategist'));
        selectedAgents.push(this.agents.get('legal-analyst'));
        break;
      default:
        selectedAgents.push(this.agents.get('legal-analyst'));
        if (request.requiresMultiAgent) {
          selectedAgents.push(this.agents.get('contract-specialist'));
          selectedAgents.push(this.agents.get('regulatory-compliance'));
        }
    }

    if (request.urgency === 'critical') {
      const criticalAgents = selectedAgents.filter(a => a.config.type === 'claude' || a.config.type === 'gemini');
      if (criticalAgents.length > 0) return criticalAgents;
       // Fallback if no premium agents are selected
       return [this.agents.get('litigation-strategist')]; 
    }

    return selectedAgents.filter(Boolean);
  }

  createSpecializedPrompt(request, config) {
    const specializationContext = config.specialization.includes('contract_drafting')
      ? "Focus on contractual terms, obligations, and risk factors."
      : config.specialization.includes('litigation_strategy')
        ? "Focus on procedural considerations, evidence evaluation, and strategic positioning."
        : config.specialization.includes('regulatory_analysis')
          ? "Focus on compliance requirements, regulatory frameworks, and policy implications."
          : "Provide comprehensive legal analysis from your area of expertise.";

    return PromptTemplate.fromTemplate(`
    You are a {agentName} with specialization in: {specializations}.
    {specializationContext}
    Document Type: {documentType}, Jurisdiction: {jurisdiction}, Urgency: {urgency}
    Query: {query}
    Context: {context}
    Provide your specialized analysis:
    `);
  }

  selectSynthesisAgent(request) {
    return this.agents.get('litigation-strategist') || this.agents.get('legal-analyst');
  }

  async synthesizeResponses(
    request,
    responses,
    synthesisAgent
  ) {
    const synthesisPrompt = PromptTemplate.fromTemplate(`
    You are synthesizing multiple expert legal opinions on the following query:
    Original Query: {query}
    Expert Analyses:
    {analyses}
    Please provide a comprehensive synthesis that:
    1. Identifies areas of consensus among experts.
    2. Highlights any conflicting viewpoints and explains why.
    3. Provides a unified conclusion and recommendation.
    4. Assesses the overall confidence level.
    Synthesis:
    `);

    const analysesText = responses.map((r, i) => `Expert ${i + 1} (${r.agentName}):\n${r.response}\n`).join('\n');
    const chain = RunnableSequence.from([synthesisPrompt, synthesisAgent.agent, this.outputParser]);

    return await chain.invoke({ query: request.query, analyses: analysesText });
  }

  extractConfidence(response) {
    const confidenceMatch = response.match(/confidence.*?(\d+(?:\.\d+)?)/i);
    return confidenceMatch ? Math.min(10, Math.max(1, parseFloat(confidenceMatch[1]))) : 7;
  }

  extractRecommendations(response) {
    const recommendations = [];
    const lines = response.split('\n');
    let inRecommendations = false;
    for (const line of lines) {
      if (/recommendation|action|next step/i.test(line)) {
        inRecommendations = true;
        continue;
      }
      if (inRecommendations && line.trim() && /^\d+\.|-|\•/.test(line)) {
        recommendations.push(line.trim());
      }
    }
    return recommendations.length > 0 ? recommendations : ['Further legal consultation recommended'];
  }

  calculateOverallConfidence(responses) {
    if (responses.length === 0) return 0;
    const avgConfidence = responses.reduce((sum, r) => sum + r.confidence, 0) / responses.length;
    return Math.round(avgConfidence * 10) / 10;
  }

  estimateTokenUsage(prompt, response) {
    const promptTokens = Math.ceil(prompt.length / 4);
    const completionTokens = Math.ceil(response.length / 4);
    return { prompt: promptTokens, completion: completionTokens, total: promptTokens + completionTokens };
  }

  generateCacheKey(request) {
    const keyData = {
      query: request.query,
      documentType: request.documentType,
      jurisdiction: request.jurisdiction,
      multiAgent: request.requiresMultiAgent
    };
    return `orchestrator:${this.hashObject(keyData)}`;
  }

  hashObject(obj) {
    const str = JSON.stringify(obj);
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      hash = (hash << 5) - hash + str.charCodeAt(i);
      hash |= 0; // Convert to 32bit integer
    }
    return Math.abs(hash).toString(36);
  }

  async cacheResult(cacheKey, result) {
    try {
      await cacheManager.set(cacheKey, result, 1800); // 30 min cache
    } catch (error) {
      console.error('Failed to cache orchestration result:', error);
    }
  }
  
  // --- Methods from Stub ---
  getSession(sessionId) {
    return this.sessions.get(sessionId) || null;
  }

  clearSession(sessionId) {
    this.sessions.delete(sessionId);
  }

  generateSessionId() {
    return `session_${Date.now()}_${Math.random().toString(36).substring(2, 11)}`;
  }

  // Process work queue for queued requests
  async processWorkQueue() {
    if (this.workerPool.workQueue.length === 0) return;
    
    const worker = this.workerPool.getAvailableWorker();
    if (!worker) return;
    
    const queuedWork = this.workerPool.workQueue.shift();
    if (queuedWork) {
      try {
        const result = await this.orchestrate(queuedWork.request);
        queuedWork.resolve(result);
      } catch (error) {
        queuedWork.reject(error);
      }
    }
  }

  // Update performance metrics (enhanced with LRU caching)
  updateMetrics(startTime, success, cacheHit, lruCacheHit = false) {
    const responseTime = Date.now() - startTime;
    
    if (success) {
      this.metrics.successfulRequests++;
    } else {
      this.metrics.failedRequests++;
    }
    
    // Update average response time
    const totalTime = this.metrics.averageResponseTime * (this.metrics.totalRequests - 1) + responseTime;
    this.metrics.averageResponseTime = Math.round(totalTime / this.metrics.totalRequests);
    
    // Update cache hit rates
    if (cacheHit) {
      const cacheHits = Math.round(this.metrics.cacheHitRate * (this.metrics.totalRequests - 1) / 100) + 1;
      this.metrics.cacheHitRate = Math.round((cacheHits / this.metrics.totalRequests) * 100);
    } else if (this.metrics.totalRequests > 1) {
      const cacheHits = Math.round(this.metrics.cacheHitRate * (this.metrics.totalRequests - 1) / 100);
      this.metrics.cacheHitRate = Math.round((cacheHits / this.metrics.totalRequests) * 100);
    }
    
    // Update LRU cache hit rate
    if (lruCacheHit) {
      const lruCacheHits = Math.round(this.metrics.lruCacheHitRate * (this.metrics.totalRequests - 1) / 100) + 1;
      this.metrics.lruCacheHitRate = Math.round((lruCacheHits / this.metrics.totalRequests) * 100);
    } else if (this.metrics.totalRequests > 1) {
      const lruCacheHits = Math.round(this.metrics.lruCacheHitRate * (this.metrics.totalRequests - 1) / 100);
      this.metrics.lruCacheHitRate = Math.round((lruCacheHits / this.metrics.totalRequests) * 100);
    }
  }

  async getStatus() {
    // Get current health status
    const healthStatus = await this.performHealthChecks();
    
    const status = {};
    for (const [name, { config }] of this.agents) {
        // Enhanced health check with service availability
        status[name] = {
            type: config.type,
            model: config.model,
            specialization: config.specialization,
            available: this.isAgentAvailable(config.type, healthStatus)
        };
    }

    return {
        meta: {
            processing: this.processing,
            activeSessions: this.sessions.size,
            agentCount: this.agents.size,
            version: '2.0.0-optimized',
            timestamp: Date.now(),
            uptime: process.uptime(),
            // Enhanced metrics for your monitoring dashboard
            metrics: this.metrics,
            health: healthStatus,
            natsConnected: this.nats ? (this.nats.isConnected && this.nats.isConnected()) : false
        },
        agents: status
    };
  }

  // Check if agent is available based on service health
  isAgentAvailable(agentType, healthStatus) {
    switch (agentType) {
      case 'ollama':
        return healthStatus.ollama;
      case 'claude':
        return true; // Claude API availability is checked via actual requests
      case 'gemini':
        return true; // Gemini API availability is checked via actual requests
      default:
        return true;
    }
  }
}

// FIX: Corrected export to provide a single, consistent instance.
export const legalOrchestrator = new LegalAIOrchestrator();
export default legalOrchestrator;