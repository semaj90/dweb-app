// Did You Mean Client - Legal AI Suggestion System
// Ultra-low latency QUIC-based suggestion client

export interface Suggestion {
	text: string;
	type: 'typo' | 'semantic' | 'completion' | 'graph' | 'synonym';
	confidence: number;
	legal_context?: string;
	practice_area?: string;
	icon: string;
}

export interface SuggestionResponse {
	suggestions: Suggestion[];
	query: string;
	processing_time_ms: number;
	cache_hit: boolean;
	stream_id?: string;
	graph_traversal_ms?: number;
}

export interface SuggestionContext {
	case_type?: string;
	document_type?: string;
	practice_area?: string;
	jurisdiction?: string;
	user_role?: string;
}

export class DidYouMeanClient {
	private baseUrl: string;
	private cache: Map<string, { data: SuggestionResponse; timestamp: number }> = new Map();
	private readonly cacheTimeout = 5 * 60 * 1000; // 5 minutes

	constructor(baseUrl: string = '') {
		this.baseUrl = baseUrl;
	}

	/**
	 * Get suggestions for a query
	 */
	async getSuggestions(
		query: string,
		intent: string = 'legal_research',
		format: 'json' | 'binary' = 'json'
	): Promise<SuggestionResponse> {
		// Check local cache first
		const cacheKey = `${query.toLowerCase().trim()}_${intent}`;
		const cached = this.cache.get(cacheKey);
		if (cached && Date.now() - cached.timestamp < this.cacheTimeout) {
			return { ...cached.data, cache_hit: true };
		}

		const params = new URLSearchParams({
			q: query,
			intent,
			format
		});

		const response = await fetch(
			`${this.baseUrl}/api/suggest/did-you-mean?${params.toString()}`,
			{
				method: 'GET',
				headers: {
					'Accept': format === 'binary' ? 'application/octet-stream' : 'application/json'
				}
			}
		);

		if (!response.ok) {
			throw new Error(`Suggestion API error: ${response.status} ${response.statusText}`);
		}

		let result: SuggestionResponse;

		if (format === 'binary') {
			const buffer = await response.arrayBuffer();
			const decoder = new TextDecoder();
			const jsonString = decoder.decode(buffer);
			result = JSON.parse(jsonString);
		} else {
			result = await response.json();
		}

		// Cache the result
		this.cache.set(cacheKey, { data: result, timestamp: Date.now() });

		return result;
	}

	/**
	 * Get contextual suggestions with additional parameters
	 */
	async getContextualSuggestions(
		query: string,
		context: SuggestionContext,
		maxSuggestions: number = 8
	): Promise<SuggestionResponse> {
		const response = await fetch(`${this.baseUrl}/api/suggest/did-you-mean`, {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
			},
			body: JSON.stringify({
				query,
				context,
				practice_area: context.practice_area,
				max_suggestions: maxSuggestions
			})
		});

		if (!response.ok) {
			const errorText = await response.text();
			throw new Error(`Contextual suggestion error: ${errorText}`);
		}

		return await response.json();
	}

	/**
	 * Clear suggestion cache
	 */
	async clearCache(pattern?: string): Promise<{ cleared: number; pattern?: string }> {
		const params = new URLSearchParams();
		if (pattern) {
			params.set('pattern', pattern);
		}

		const response = await fetch(
			`${this.baseUrl}/api/suggest/did-you-mean?${params.toString()}`,
			{
				method: 'DELETE',
			}
		);

		if (!response.ok) {
			throw new Error(`Cache clear error: ${response.status}`);
		}

		// Clear local cache as well
		if (pattern) {
			let cleared = 0;
			for (const [key] of this.cache) {
				if (key.includes(pattern)) {
					this.cache.delete(key);
					cleared++;
				}
			}
		} else {
			this.cache.clear();
		}

		return await response.json();
	}

	/**
	 * Real-time suggestion stream for live typing
	 */
	async *streamSuggestions(
		queryStream: AsyncIterable<string>,
		context?: SuggestionContext
	): AsyncGenerator<SuggestionResponse> {
		for await (const query of queryStream) {
			if (query.trim().length === 0) continue;

			try {
				const suggestions = context
					? await this.getContextualSuggestions(query, context)
					: await this.getSuggestions(query);

				yield suggestions;
			} catch (error) {
				console.error('Stream suggestion error:', error);
				// Continue streaming despite errors
			}
		}
	}

	/**
	 * Batch suggestion processing
	 */
	async getBatchSuggestions(
		queries: string[],
		context?: SuggestionContext
	): Promise<Map<string, SuggestionResponse>> {
		const results = new Map<string, SuggestionResponse>();

		// Process in parallel with concurrency limit
		const concurrencyLimit = 5;
		const chunks = this.chunkArray(queries, concurrencyLimit);

		for (const chunk of chunks) {
			const promises = chunk.map(async (query) => {
				try {
					const suggestions = context
						? await this.getContextualSuggestions(query, context)
						: await this.getSuggestions(query);
					return { query, suggestions };
				} catch (error) {
					console.error(`Batch suggestion error for query "${query}":`, error);
					return null;
				}
			});

			const chunkResults = await Promise.all(promises);

			for (const result of chunkResults) {
				if (result) {
					results.set(result.query, result.suggestions);
				}
			}
		}

		return results;
	}

	/**
	 * Get suggestion metrics
	 */
	getSuggestionMetrics(): {
		cache_size: number;
		cache_hit_rate: number;
		average_processing_time: number;
	} {
		const cacheSize = this.cache.size;
		let totalProcessingTime = 0;
		let cacheHits = 0;

		for (const [_, { data }] of this.cache) {
			totalProcessingTime += data.processing_time_ms;
			if (data.cache_hit) {
				cacheHits++;
			}
		}

		const avgProcessingTime = cacheSize > 0 ? totalProcessingTime / cacheSize : 0;
		const cacheHitRate = cacheSize > 0 ? (cacheHits / cacheSize) * 100 : 0;

		return {
			cache_size: cacheSize,
			cache_hit_rate: cacheHitRate,
			average_processing_time: avgProcessingTime
		};
	}

	/**
	 * Filter suggestions by confidence threshold
	 */
	filterByConfidence(
		suggestions: Suggestion[],
		minConfidence: number = 0.5
	): Suggestion[] {
		return suggestions.filter(suggestion => suggestion.confidence >= minConfidence);
	}

	/**
	 * Group suggestions by type
	 */
	groupSuggestionsByType(suggestions: Suggestion[]): Record<string, Suggestion[]> {
		return suggestions.reduce((groups, suggestion) => {
			const type = suggestion.type;
			if (!groups[type]) {
				groups[type] = [];
			}
			groups[type].push(suggestion);
			return groups;
		}, {} as Record<string, Suggestion[]>);
	}

	/**
	 * Get suggestions for legal practice areas
	 */
	async getLegalPracticeAreaSuggestions(
		query: string,
		practiceAreas: string[] = ['corporate', 'criminal', 'civil', 'family']
	): Promise<Record<string, SuggestionResponse>> {
		const results: Record<string, SuggestionResponse> = {};

		for (const area of practiceAreas) {
			try {
				const context: SuggestionContext = { practice_area: area };
				const suggestions = await this.getContextualSuggestions(query, context);
				results[area] = suggestions;
			} catch (error) {
				console.error(`Error getting suggestions for practice area ${area}:`, error);
			}
		}

		return results;
	}

	/**
	 * Debounced suggestion fetching for real-time typing
	 */
	private debounceTimer: NodeJS.Timeout | null = null;

	getDebouncedSuggestions(
		query: string,
		callback: (suggestions: SuggestionResponse) => void,
		delay: number = 150,
		context?: SuggestionContext
	): void {
		if (this.debounceTimer) {
			clearTimeout(this.debounceTimer);
		}

		this.debounceTimer = setTimeout(async () => {
			try {
				const suggestions = context
					? await this.getContextualSuggestions(query, context)
					: await this.getSuggestions(query);
				callback(suggestions);
			} catch (error) {
				console.error('Debounced suggestion error:', error);
			}
		}, delay);
	}

	/**
	 * Utility function to chunk arrays
	 */
	private chunkArray<T>(array: T[], chunkSize: number): T[][] {
		const chunks: T[][] = [];
		for (let i = 0; i < array.length; i += chunkSize) {
			chunks.push(array.slice(i, i + chunkSize));
		}
		return chunks;
	}

	/**
	 * Clean up resources
	 */
	cleanup(): void {
		if (this.debounceTimer) {
			clearTimeout(this.debounceTimer);
			this.debounceTimer = null;
		}
		this.cache.clear();
	}
}

// Default client instance
export const didYouMeanClient = new DidYouMeanClient();

// Utility functions for suggestion processing
export const SuggestionUtils = {
	/**
	 * Format suggestion for display
	 */
	formatSuggestion(suggestion: Suggestion): string {
		return `${suggestion.icon} ${suggestion.text}`;
	},

	/**
	 * Get suggestion color based on type
	 */
	getSuggestionColor(type: Suggestion['type']): string {
		const colors = {
			typo: '#f59e0b',      // amber
			semantic: '#3b82f6',   // blue
			completion: '#10b981', // emerald
			graph: '#8b5cf6',      // violet
			synonym: '#06b6d4'     // cyan
		};
		return colors[type] || '#6b7280'; // default gray
	},

	/**
	 * Calculate suggestion relevance score
	 */
	calculateRelevanceScore(
		suggestion: Suggestion,
		query: string,
		context?: SuggestionContext
	): number {
		let score = suggestion.confidence;

		// Boost score if suggestion matches query closely
		const queryWords = query.toLowerCase().split(' ');
		const suggestionWords = suggestion.text.toLowerCase().split(' ');
		const commonWords = queryWords.filter(word => suggestionWords.includes(word));
		const commonWordBonus = (commonWords.length / queryWords.length) * 0.2;

		score += commonWordBonus;

		// Boost score for practice area match
		if (context?.practice_area && suggestion.practice_area === context.practice_area) {
			score += 0.1;
		}

		// Boost score for legal context relevance
		if (suggestion.legal_context && context?.case_type) {
			const legalContextWords = suggestion.legal_context.toLowerCase().split(' ');
			if (legalContextWords.some(word => context.case_type?.toLowerCase().includes(word))) {
				score += 0.15;
			}
		}

		return Math.min(1, score); // Cap at 1.0
	},

	/**
	 * Sort suggestions by relevance
	 */
	sortByRelevance(
		suggestions: Suggestion[],
		query: string,
		context?: SuggestionContext
	): Suggestion[] {
		return [...suggestions].sort((a, b) => {
			const scoreA = this.calculateRelevanceScore(a, query, context);
			const scoreB = this.calculateRelevanceScore(b, query, context);
			return scoreB - scoreA;
		});
	}
};