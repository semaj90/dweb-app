// @ts-nocheck
/**
 * Qdrant Vector Database Service
 * High-performance vector search and auto-tagging integration
 */

import { QdrantClient } from '@qdrant/js-client-rest';
import type { PointStruct, Filter, SearchRequest } from '@qdrant/js-client-rest';
import { writable, type Writable } from 'svelte/store';

// Qdrant configuration
const QDRANT_HOST = 'http://localhost:6333';
const QDRANT_COLLECTIONS = {
	documents: 'legal_documents',
	users: 'user_profiles', 
	activities: 'user_activities',
	tags: 'semantic_tags'
} as const;

// Types
export interface DocumentVector {
	id: string;
	content: string;
	embedding: number[];
	metadata: DocumentMetadata;
	tags: string[];
	timestamp: number;
}

export interface DocumentMetadata {
	title: string;
	type: 'legal_document' | 'case_law' | 'regulation' | 'contract' | 'brief';
	author?: string;
	date?: string;
	jurisdiction?: string;
	practice_area?: string;
	confidence_score?: number;
	source_path?: string;
	page_count?: number;
	language?: string;
}

// Enhanced Legal Document Types for RAG Integration
export interface LegalDocumentVector extends DocumentVector {
	// Legal-specific fields
	caseId: string;
	caseType: 'contract' | 'litigation' | 'compliance' | 'regulatory';
	legalJurisdiction: 'federal' | 'state' | 'local' | 'international';
	summary?: string;
	
	// AI analysis results
	legalEntities: {
		parties: string[];
		dates: string[];
		monetary: string[];
		clauses: string[];
		jurisdictions: string[];
		caseTypes: string[];
	};
	riskScore: number; // 0-100
	confidenceScore: number; // 0-1
	
	// Legal precedent information
	legalPrecedent: boolean;
	precedentialValue?: 'binding' | 'persuasive' | 'non_precedential';
	
	// Processing metadata
	processingStatus: 'pending' | 'processing' | 'completed' | 'error';
	aiModelVersion?: string;
	processedAt?: number;
}

export interface SearchResult {
	id: string;
	score: number;
	payload: DocumentVector;
	highlights?: string[];
}

export interface TagPrediction {
	tag: string;
	confidence: number;
	category: 'practice_area' | 'document_type' | 'legal_concept' | 'jurisdiction';
	source: 'llm' | 'pattern_matching' | 'manual';
}

// Qdrant Service Class
export class QdrantService {
	public client: QdrantClient;
	private isConnected = false;
	
	// Reactive stores for UI integration
	public connectionStatus$: Writable<'connected' | 'disconnected' | 'connecting' | 'error'> = writable('disconnected');
	public collections$: Writable<string[]> = writable([]);
	public searchResults$: Writable<SearchResult[]> = writable([]);

	constructor() {
		this.client = new QdrantClient({ 
			host: QDRANT_HOST,
			port: 6333,
			https: false
		});
		this.initialize();
	}

	private async initialize(): Promise<void> {
		try {
			this.connectionStatus$.set('connecting');
			
			// Test connection
			await this.client.getCollections();
			this.isConnected = true;
			this.connectionStatus$.set('connected');
			
			// Initialize collections
			await this.initializeCollections();
			
			// Update collections list
			const collections = await this.client.getCollections();
			this.collections$.set(collections.collections.map((c: any) => c.name));
			
			console.log('🔗 Qdrant connected successfully');
			
		} catch (error) {
			console.error('❌ Qdrant connection failed:', error);
			this.connectionStatus$.set('error');
			this.isConnected = false;
		}
	}

	private async initializeCollections(): Promise<void> {
		const collections = Object.values(QDRANT_COLLECTIONS);
		
		for (const collectionName of collections) {
			try {
				await this.client.getCollection(collectionName);
				console.log(`✅ Collection '${collectionName}' exists`);
			} catch {
				console.log(`🔧 Creating collection '${collectionName}'`);
				await this.createCollection(collectionName);
			}
		}
	}

	private async createCollection(name: string): Promise<void> {
		const vectorSize = 384; // Nomic-embed-text dimensions
		
		await this.client.createCollection(name, {
			vectors: {
				size: vectorSize,
				distance: 'Cosine'
			},
			optimizers_config: {
				default_segment_number: 2,
				max_segment_size: 20000,
				memmap_threshold: 20000,
				indexing_threshold: 10000,
				flush_interval_sec: 10,
				max_optimization_threads: 2
			},
			hnsw_config: {
				m: 16,
				ef_construct: 200,
				full_scan_threshold: 10000,
				max_indexing_threads: 2
			}
		});
	}

	// Document operations
	public async addDocument(document: Omit<DocumentVector, 'id' | 'timestamp'>): Promise<string> {
		if (!this.isConnected) throw new Error('Qdrant not connected');

		const documentId = `doc_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
		
		// Auto-generate tags before storing
		const autoTags = await this.generateAutoTags(document.content, document.metadata);
		
		const point: PointStruct = {
			id: documentId,
			vector: document.embedding,
			payload: {
				...document,
				id: documentId,
				timestamp: Date.now(),
				tags: [...document.tags, ...autoTags.map((t: any) => t.tag)]
			}
		};

		await this.client.upsert(QDRANT_COLLECTIONS.documents, {
			wait: true,
			points: [point]
		});

		// Store tag predictions separately for analysis
		await this.storeTagPredictions(documentId, autoTags);

		console.log(`📄 Document '${documentId}' added with ${autoTags.length} auto-tags`);
		return documentId;
	}

	public async searchDocuments(
		query: string,
		embedding: number[],
		options: {
			limit?: number;
			filter?: Filter;
			scoreThreshold?: number;
			withPayload?: boolean;
		} = {}
	): Promise<SearchResult[]> {
		if (!this.isConnected) throw new Error('Qdrant not connected');

		const searchRequest: SearchRequest = {
			vector: embedding,
			limit: options.limit || 10,
			score_threshold: options.scoreThreshold || 0.7,
			with_payload: options.withPayload !== false,
			filter: options.filter
		};

		const results = await this.client.search(QDRANT_COLLECTIONS.documents, searchRequest);
		
		const searchResults: SearchResult[] = results.map((result: any) => ({
		id: result.id.toString(),
		score: result.score,
		payload: result.payload as DocumentVector & { [key: string]: unknown },
		highlights: this.extractHighlights(query, result.payload as DocumentVector & { [key: string]: unknown })
		}));

		this.searchResults$.set(searchResults);
		return searchResults;
	}

	// Enhanced semantic search with auto-filtering
	public async semanticSearch(
		query: string,
		embedding: number[],
		options: {
			practiceArea?: string;
			documentType?: string;
			jurisdiction?: string;
			dateRange?: { start: string; end: string };
			limit?: number;
		} = {}
	): Promise<SearchResult[]> {
		// Build dynamic filter based on options
		const filter: Filter = {
			must: []
		};

		if (options.practiceArea) {
			filter.must!.push({
				key: 'metadata.practice_area',
				match: { value: options.practiceArea }
			});
		}

		if (options.documentType) {
			filter.must!.push({
				key: 'metadata.type',
				match: { value: options.documentType }
			});
		}

		if (options.jurisdiction) {
			filter.must!.push({
				key: 'metadata.jurisdiction',
				match: { value: options.jurisdiction }
			});
		}

		if (options.dateRange) {
			filter.must!.push({
				key: 'metadata.date',
				range: {
					gte: options.dateRange.start,
					lte: options.dateRange.end
				}
			});
		}

		return this.searchDocuments(query, embedding, {
			filter: filter.must!.length > 0 ? filter : undefined,
			limit: options.limit
		});
	}

	// Legal-specific search with enhanced RAG integration
	public async searchLegalDocuments(
		queryVector: number[],
		options: {
			caseType?: 'contract' | 'litigation' | 'compliance' | 'regulatory';
			jurisdiction?: 'federal' | 'state' | 'local' | 'international';
			riskThreshold?: number;
			requirePrecedent?: boolean;
			tags?: string[];
			dateRange?: { from: string; to: string };
			limit?: number;
			scoreThreshold?: number;
		} = {}
	): Promise<SearchResult[]> {
		const filter: Filter = { must: [] };

		// Apply legal-specific filters
		if (options.caseType) {
			filter.must!.push({
				key: 'caseType',
				match: { value: options.caseType }
			});
		}

		if (options.jurisdiction) {
			filter.must!.push({
				key: 'legalJurisdiction',
				match: { value: options.jurisdiction }
			});
		}

		if (options.riskThreshold !== undefined) {
			filter.must!.push({
				key: 'riskScore',
				range: { gte: options.riskThreshold }
			});
		}

		if (options.requirePrecedent) {
			filter.must!.push({
				key: 'legalPrecedent',
				match: { value: true }
			});
		}

		if (options.tags && options.tags.length > 0) {
			const tagFilters = options.tags.map((tag: any) => ({
				key: 'tags',
				match: { value: tag }
			}));
			if (!filter.should) filter.should = [];
			filter.should.push(...tagFilters);
		}

		if (options.dateRange) {
			filter.must!.push({
				key: 'processedAt',
				range: { 
					gte: new Date(options.dateRange.from).getTime(),
					lte: new Date(options.dateRange.to).getTime()
				}
			});
		}

		const searchRequest: SearchRequest = {
			vector: queryVector,
			limit: options.limit || 10,
			score_threshold: options.scoreThreshold || 0.7,
			with_payload: true,
			filter: filter.must!.length > 0 || filter.should?.length ? filter : undefined
		};

		const results = await this.client.search(QDRANT_COLLECTIONS.documents, searchRequest);
		
		return results.map((result: any) => ({
		id: result.id.toString(),
		score: result.score,
		payload: result.payload as LegalDocumentVector & Record<string, unknown>,
		highlights: this.extractLegalHighlights(result.payload as LegalDocumentVector & Record<string, unknown>)
		}));
	}

	// Add legal document with enhanced processing
	public async addLegalDocument(document: Omit<LegalDocumentVector, 'id' | 'timestamp'>): Promise<string> {
		if (!this.isConnected) throw new Error('Qdrant not connected');

		const documentId = `legal_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
		
		// Enhanced auto-tagging for legal documents
		const autoTags = await this.generateLegalAutoTags(document.content, document);
		
		const point: PointStruct = {
			id: documentId,
			vector: document.embedding,
			payload: {
				...document,
				id: documentId,
				timestamp: Date.now(),
				tags: [...document.tags, ...autoTags.map((t: any) => t.tag)],
				processedAt: Date.now(),
				processingStatus: 'completed'
			}
		};

		await this.client.upsert(QDRANT_COLLECTIONS.documents, {
			wait: true,
			points: [point]
		});

		// Store enhanced tag predictions for legal analysis
		await this.storeLegalTagPredictions(documentId, autoTags);

		console.log(`⚖️ Legal document '${documentId}' added with ${autoTags.length} auto-tags`);
		return documentId;
	}

	// Enhanced legal auto-tagging
	private async generateLegalAutoTags(content: string, document: LegalDocumentVector): Promise<TagPrediction[]> {
		try {
			const response = await fetch('http://localhost:11434/api/generate', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					model: 'gemma3-legal',
					prompt: this.buildLegalTaggingPrompt(content, document),
					stream: false,
					format: 'json'
				})
			});

			if (!response.ok) {
				throw new Error(`Legal tagging LLM request failed: ${response.statusText}`);
			}

			const data = await response.json();
			const taggingResult = JSON.parse(data.response);

			return this.parseLegalLLMTags(taggingResult);
			
		} catch (error) {
			console.warn('⚠️ Legal auto-tagging failed, using fallback:', error);
			return this.fallbackLegalTags(content, document);
		}
	}

	private buildLegalTaggingPrompt(content: string, document: LegalDocumentVector): string {
		return `
Analyze this legal document for advanced semantic tagging. Focus on:
1. Legal practice areas (contract law, tort law, criminal law, etc.)
2. Case types (${document.caseType})
3. Jurisdiction-specific concepts (${document.legalJurisdiction})
4. Risk indicators (liability, compliance, regulatory)
5. Legal precedents and citations
6. Legal entities (parties, courts, statutes)

Document Context:
- Case Type: ${document.caseType}
- Jurisdiction: ${document.legalJurisdiction}
- Risk Score: ${document.riskScore}
- Legal Precedent: ${document.legalPrecedent}
- Content Sample: ${content.substring(0, 1500)}...

Return a JSON object with this enhanced structure:
{
  "practice_areas": ["contract_law", "corporate_law"],
  "legal_concepts": ["liability", "damages", "jurisdiction"],
  "case_types": ["breach_of_contract", "negligence"],
  "jurisdictions": ["federal", "state_specific"],
  "risk_indicators": ["high_liability", "regulatory_compliance"],
  "legal_entities": ["plaintiff", "defendant", "court"],
  "precedent_relevance": ["binding", "persuasive"],
  "document_characteristics": ["complex", "time_sensitive"],
  "confidence_scores": {"contract_law": 0.95, "liability": 0.87}
}`;
	}

	private parseLegalLLMTags(llmResult: any): TagPrediction[] {
		const tags: TagPrediction[] = [];

		const legalCategories = [
			{ key: 'practice_areas', category: 'practice_area' as const },
			{ key: 'legal_concepts', category: 'legal_concept' as const },
			{ key: 'case_types', category: 'document_type' as const },
			{ key: 'jurisdictions', category: 'jurisdiction' as const },
			{ key: 'risk_indicators', category: 'legal_concept' as const },
			{ key: 'legal_entities', category: 'legal_concept' as const },
			{ key: 'precedent_relevance', category: 'legal_concept' as const },
			{ key: 'document_characteristics', category: 'document_type' as const }
		];

		for (const { key, category } of legalCategories) {
			const categoryTags = llmResult[key] || [];
			for (const tag of categoryTags) {
				tags.push({
					tag,
					confidence: llmResult.confidence_scores?.[tag] || 0.8,
					category,
					source: 'llm'
				});
			}
		}

		return tags;
	}

	private fallbackLegalTags(content: string, document: LegalDocumentVector): TagPrediction[] {
		const tags: TagPrediction[] = [];
		
		// Enhanced legal pattern matching
		const legalPatterns = {
			'contract_law': /\b(agreement|contract|terms|conditions|breach|performance)\b/i,
			'tort_law': /\b(negligence|liability|damages|injury|duty|care)\b/i,
			'corporate_law': /\b(corporation|LLC|board|director|shareholder|merger)\b/i,
			'intellectual_property': /\b(patent|trademark|copyright|IP|trade\s*secret)\b/i,
			'employment_law': /\b(employee|employer|workplace|discrimination|termination)\b/i,
			'litigation': /\b(lawsuit|court|judge|trial|motion|discovery)\b/i,
			'compliance': /\b(regulation|compliance|audit|violation|penalty)\b/i,
			'criminal_law': /\b(criminal|felony|misdemeanor|prosecution|defense)\b/i
		};

		for (const [tag, pattern] of Object.entries(legalPatterns)) {
			if (pattern.test(content)) {
				tags.push({
					tag,
					confidence: 0.7,
					category: 'practice_area',
					source: 'pattern_matching'
				});
			}
		}

		// Add context-based tags
		tags.push({
			tag: document.caseType,
			confidence: 0.9,
			category: 'document_type',
			source: 'manual'
		});

		tags.push({
			tag: document.legalJurisdiction,
			confidence: 0.9,
			category: 'jurisdiction',
			source: 'manual'
		});

		// Risk-based tagging
		if (document.riskScore > 70) {
			tags.push({
				tag: 'high_risk',
				confidence: 0.8,
				category: 'legal_concept',
				source: 'pattern_matching'
			});
		}

		return tags;
	}

	private async storeLegalTagPredictions(documentId: string, predictions: TagPrediction[]): Promise<void> {
		const points: PointStruct[] = predictions.map((prediction, index) => ({
			id: `legal_tag_${documentId}_${index}`,
			vector: new Array(384).fill(0), // Placeholder vector for tags
			payload: {
				document_id: documentId,
				...prediction,
				timestamp: Date.now(),
				legal_context: true
			}
		}));

		await this.client.upsert(QDRANT_COLLECTIONS.tags, {
			wait: true,
			points
		});
	}

	private extractLegalHighlights(document: LegalDocumentVector): string[] {
		const highlights: string[] = [];
		
		// Extract key legal entities as highlights
		if (document.legalEntities) {
		document.legalEntities.parties?.slice(0, 2).forEach((party: any) => {
		highlights.push(`Party: ${party}`);
		});
		
		document.legalEntities.monetary?.slice(0, 2).forEach((amount: any) => {
		highlights.push(`Amount: ${amount}`);
		});
		
		document.legalEntities.clauses?.slice(0, 2).forEach((clause: any) => {
		highlights.push(`Clause: ${clause}`);
		});
		}

		// Add case type and jurisdiction
		highlights.push(`${document.caseType} (${document.legalJurisdiction})`);
		
		// Add risk indicator
		if (document.riskScore > 50) {
			highlights.push(`Risk Score: ${document.riskScore}/100`);
		}

		return highlights.slice(0, 5); // Return top 5 legal highlights
	}

	// Auto-tagging with LLM integration
	private async generateAutoTags(content: string, metadata: DocumentMetadata): Promise<TagPrediction[]> {
		try {
			// Call Ollama for intelligent tagging
			const response = await fetch('http://localhost:11434/api/generate', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					model: 'gemma3-legal',
					prompt: this.buildTaggingPrompt(content, metadata),
					stream: false,
					format: 'json'
				})
			});

			if (!response.ok) {
				throw new Error(`Tagging LLM request failed: ${response.statusText}`);
			}

			const data = await response.json();
			const taggingResult = JSON.parse(data.response);

			return this.parseLLMTags(taggingResult);
			
		} catch (error) {
			console.warn('⚠️ Auto-tagging failed, using fallback:', error);
			return this.fallbackTags(content, metadata);
		}
	}

	private buildTaggingPrompt(content: string, metadata: DocumentMetadata): string {
		return `
Analyze this legal document and generate semantic tags. Focus on:
1. Practice areas (litigation, corporate, IP, etc.)  
2. Document types (contract, brief, motion, etc.)
3. Legal concepts (jurisdiction, liability, compliance, etc.)
4. Key entities (parties, statutes, cases, etc.)

Document Type: ${metadata.type}
Content Sample: ${content.substring(0, 1000)}...

Return a JSON object with this structure:
{
  "practice_areas": ["tag1", "tag2"],
  "legal_concepts": ["concept1", "concept2"], 
  "entities": ["entity1", "entity2"],
  "document_characteristics": ["urgent", "complex", "routine"],
  "confidence_scores": {"tag1": 0.95, "tag2": 0.87}
}`;
	}

	private parseLLMTags(llmResult: any): TagPrediction[] {
		const tags: TagPrediction[] = [];

		// Parse different tag categories
		const categories = [
			{ key: 'practice_areas', category: 'practice_area' as const },
			{ key: 'legal_concepts', category: 'legal_concept' as const },
			{ key: 'entities', category: 'legal_concept' as const },
			{ key: 'document_characteristics', category: 'document_type' as const }
		];

		for (const { key, category } of categories) {
			const categoryTags = llmResult[key] || [];
			for (const tag of categoryTags) {
				tags.push({
					tag,
					confidence: llmResult.confidence_scores?.[tag] || 0.8,
					category,
					source: 'llm'
				});
			}
		}

		return tags;
	}

	private fallbackTags(content: string, metadata: DocumentMetadata): TagPrediction[] {
		const tags: TagPrediction[] = [];
		
		// Basic pattern matching
		const patterns = {
			'contract': /\b(agreement|contract|terms|conditions)\b/i,
			'litigation': /\b(lawsuit|court|judge|trial|motion)\b/i,
			'corporate': /\b(corporation|LLC|board|director|shareholder)\b/i,
			'intellectual_property': /\b(patent|trademark|copyright|IP)\b/i,
			'employment': /\b(employee|employer|workplace|discrimination)\b/i
		};

		for (const [tag, pattern] of Object.entries(patterns)) {
			if (pattern.test(content)) {
				tags.push({
					tag,
					confidence: 0.6,
					category: 'practice_area',
					source: 'pattern_matching'
				});
			}
		}

		// Add document type tag
		if (metadata.type) {
			tags.push({
				tag: metadata.type,
				confidence: 0.9,
				category: 'document_type',
				source: 'manual'
			});
		}

		return tags;
	}

	private async storeTagPredictions(documentId: string, predictions: TagPrediction[]): Promise<void> {
		const points: PointStruct[] = predictions.map((prediction, index) => ({
			id: `tag_${documentId}_${index}`,
			vector: new Array(384).fill(0), // Placeholder vector for tags
			payload: {
				document_id: documentId,
				...prediction,
				timestamp: Date.now()
			}
		}));

		await this.client.upsert(QDRANT_COLLECTIONS.tags, {
			wait: true,
			points
		});
	}

	private extractHighlights(query: string, document: DocumentVector): string[] {
		// Simple highlight extraction - can be enhanced with more sophisticated NLP
		const queryTerms = query.toLowerCase().split(' ');
		const content = document.content.toLowerCase();
		const highlights: string[] = [];

		for (const term of queryTerms) {
			const index = content.indexOf(term);
			if (index !== -1) {
				const start = Math.max(0, index - 50);
				const end = Math.min(content.length, index + term.length + 50);
				const highlight = document.content.substring(start, end);
				highlights.push(`...${highlight}...`);
			}
		}

		return highlights.slice(0, 3); // Return top 3 highlights
	}

	// User activity tracking
	public async logUserActivity(
		userId: string, 
		activity: {
			type: 'search' | 'view' | 'edit' | 'bookmark' | 'share';
			documentId?: string;
			query?: string;
			metadata?: Record<string, any>;
		}
	): Promise<void> {
		if (!this.isConnected) return;

		const activityId = `activity_${userId}_${Date.now()}`;
		
		// Generate embedding for the activity (for activity-based recommendations)
		const activityText = `${activity.type} ${activity.query || ''} ${activity.documentId || ''}`;
		const embedding = await this.generateEmbedding(activityText);

		const point: PointStruct = {
			id: activityId,
			vector: embedding,
			payload: {
				user_id: userId,
				timestamp: Date.now(),
				...activity
			}
		};

		await this.client.upsert(QDRANT_COLLECTIONS.activities, {
			wait: true,
			points: [point]
		});
	}

	// Utility methods
	private async generateEmbedding(text: string): Promise<number[]> {
		try {
			const response = await fetch('http://localhost:11434/api/embeddings', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					model: 'nomic-embed-text',
					prompt: text
				})
			});

			if (!response.ok) throw new Error(`Embedding failed: ${response.statusText}`);
			
			const data = await response.json();
			return data.embedding;
			
		} catch (error) {
			console.error('Embedding generation failed:', error);
			return new Array(384).fill(0); // Fallback zero vector
		}
	}

	// System utilities
	public async getCollectionInfo(collectionName: string) {
		if (!this.isConnected) throw new Error('Qdrant not connected');
		return await this.client.getCollection(collectionName);
	}

	public async getSystemHealth() {
		if (!this.isConnected) return { status: 'disconnected' };
		
		try {
			const collections = await this.client.getCollections();
			const info = await Promise.all(
				collections.collections.map(async (col) => {
					const details = await this.client.getCollection(col.name);
					return {
						name: col.name,
						points_count: details.points_count,
						status: details.status
					};
				})
			);

			return {
				status: 'healthy',
				collections: info,
				total_points: info.reduce((sum, col) => sum + (col.points_count || 0), 0)
			};
		} catch (error) {
			return { status: 'error', error: error.message };
		}
	}

	public destroy(): void {
		// Cleanup connections if needed
		this.isConnected = false;
		this.connectionStatus$.set('disconnected');
	}
}

// Singleton instance
export const qdrantService = new QdrantService();