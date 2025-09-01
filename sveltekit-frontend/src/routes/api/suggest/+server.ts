import type { RequestHandler } from './$types.js';

// "Did You Mean?" Suggestions API - SSR compatible with fuzzy search
import { json, error } from '@sveltejs/kit';
import Fuse from 'fuse.js';
import { db } from '$lib/server/db/index.js';
import { users, cases, evidence } from '$lib/server/db/schema-unified.js';
import { generateEnhancedEmbedding } from '$lib/server/ai/embeddings-enhanced.js';
import { or, ilike, sql } from 'drizzle-orm';
import { URL } from "url";

export interface Suggestion {
  label: string;
  entityId: string;
  type: 'PERSON' | 'DOCUMENT' | 'CASE' | 'EVIDENCE' | 'TAG';
  score: number;
  description: string;
  icon: string;
  tags: string[];
}

export interface SuggestResponse {
  suggestions: Suggestion[];
  correctedQuery: string;
  explanation: string;
  processingTimeMs: number;
}

// Mock data for development - replace with database queries
const mockPeople = [
  { id: '1', name: 'Sarah Johnson', email: 'sarah@law.com', role: 'attorney', specialization: 'corporate' },
  { id: '2', name: 'Michael Chen', email: 'mchen@legal.com', role: 'paralegal', specialization: 'litigation' },
  { id: '3', name: 'Emily Rodriguez', email: 'emily.r@law.com', role: 'investigator', specialization: 'evidence' },
  { id: '4', name: 'David Thompson', email: 'dthompson@legal.com', role: 'attorney', specialization: 'criminal' },
];

const mockCases = [
  { id: 'case-1', title: 'Corporate Merger Review', description: 'M&A due diligence case', status: 'active' },
  { id: 'case-2', title: 'Employment Discrimination', description: 'Workplace harassment investigation', status: 'pending' },
  { id: 'case-3', title: 'Contract Dispute Resolution', description: 'Breach of service agreement', status: 'closed' },
];

const mockDocuments = [
  { id: 'doc-1', title: 'Service Agreement Template', type: 'contract', category: 'templates' },
  { id: 'doc-2', title: 'Evidence Collection Protocol', type: 'procedure', category: 'evidence' },
  { id: 'doc-3', title: 'Legal Research Memo', type: 'memo', category: 'research' },
];

export const GET: RequestHandler = async ({ url, fetch }) => {
  const startTime = Date.now();
  const query = url.searchParams.get('q') || '';
  const limit = parseInt(url.searchParams.get('limit') || '10');
  const contextType = url.searchParams.get('context') || 'GENERAL';
  const userId = url.searchParams.get('userId') || '';

  if (!query || query.length < 2) {
    throw error(400, 'Query must be at least 2 characters');
  }

  try {
    const suggestions: Suggestion[] = [];

    // Check recommendations service first
    let useService = false;
    try {
      const serviceResponse = await fetch(`http://localhost:8105/api/suggest?q=${encodeURIComponent(query)}&limit=${limit}&context=${contextType}`);
      if (serviceResponse.ok) {
        const serviceData = await serviceResponse.json();
        useService = true;
        
        return json({
          suggestions: serviceData.suggestions || [],
          correctedQuery: serviceData.corrected_query || query,
          explanation: serviceData.explanation || 'Suggestions from recommendations service',
          processingTimeMs: Date.now() - startTime
        });
      }
    } catch (e: any) {
      console.log('Recommendations service not available, using fallback');
    }

    // Enhanced database search with fuzzy fallback
    if (!useService) {
      // Search database first
      const dbSuggestions = await searchDatabase(query, contextType, limit);
      suggestions.push(...dbSuggestions);
      
      // If not enough results from database, use fuzzy search on mock data
      if (suggestions.length < limit) {
        const fuzzySuggestions = await searchWithFuzzy(query, contextType, limit - suggestions.length);
        suggestions.push(...fuzzySuggestions);
      }
    }

    // Sort by score and limit results
    suggestions.sort((a, b) => b.score - a.score);
    const limitedSuggestions = suggestions.slice(0, limit);

    // Simple spell correction (can be enhanced)
    const correctedQuery = query.toLowerCase().trim();

    const response: SuggestResponse = {
      suggestions: limitedSuggestions,
      correctedQuery,
      explanation: `Found ${limitedSuggestions.length} suggestions for "${query}"`,
      processingTimeMs: Date.now() - startTime
    };

    return json(response);

  } catch (err: any) {
    console.error('Suggestions API error:', err);
    throw error(500, 'Failed to fetch suggestions');
  }
};

export const POST: RequestHandler = async ({ request, fetch }) => {
  const body = await request.json();
  const { query, contextType, userId, recentQueries } = body;

  // Enhanced suggestions with user context and query history
  const enhancedQuery = `${query} ${recentQueries?.join(' ') || ''}`.trim();
  
  // Use GET handler with enhanced query
  const url = new URL(`/api/suggest?q=${encodeURIComponent(enhancedQuery)}&context=${contextType}&userId=${userId}`, 'http://localhost');
  const mockRequest = new Request(url);
  
  return await GET({ 
    params: {}, 
    url: new URL(url), 
    fetch,
    request: mockRequest,
    route: { id: '/api/suggest' },
    locals: {},
    platform: undefined,
    isDataRequest: false,
    isSubRequest: false
  } as any);
};

async function searchDatabase(query: string, contextType: string, limit: number): Promise<Suggestion[]> {
  const suggestions: Suggestion[] = [];
  const queryLower = query.toLowerCase();

  try {
    // Search people in database
    if (contextType === 'PERSON' || contextType === 'GENERAL') {
      const people = await db
        .select({
          id: users.id,
          displayName: users.displayName,
          email: users.email,
          role: users.role
        })
        .from(users)
        .where(
          or(
            ilike(users.displayName, `%${queryLower}%`),
            ilike(users.email, `%${queryLower}%`),
            ilike(users.role, `%${queryLower}%`)
          )
        )
        .limit(Math.ceil(limit / 3));

      people.forEach(person => {
        const name = person.displayName || person.email || 'Unknown User';
        const similarity = calculateSimilarity(queryLower, name.toLowerCase());
        if (similarity > 0.3) {
          suggestions.push({
            label: name,
            entityId: person.id,
            type: 'PERSON',
            score: similarity,
            description: `${person.role || 'User'} - ${person.email}`,
            icon: 'user',
            tags: ['legal', 'professional', person.role || 'user'].filter(Boolean)
          });
        }
      });
    }

    // Search cases in database
    if (contextType === 'CASE' || contextType === 'GENERAL') {
      const dbCases = await db
        .select({
          id: cases.id,
          title: cases.title,
          description: cases.description,
          status: cases.status,
          caseType: cases.caseType
        })
        .from(cases)
        .where(
          or(
            ilike(cases.title, `%${queryLower}%`),
            ilike(cases.description, `%${queryLower}%`),
            ilike(cases.caseType, `%${queryLower}%`)
          )
        )
        .limit(Math.ceil(limit / 3));

      dbCases.forEach(caseItem => {
        const similarity = Math.max(
          calculateSimilarity(queryLower, caseItem.title.toLowerCase()),
          caseItem.description ? calculateSimilarity(queryLower, caseItem.description.toLowerCase()) * 0.8 : 0
        );
        if (similarity > 0.3) {
          suggestions.push({
            label: caseItem.title,
            entityId: caseItem.id,
            type: 'CASE',
            score: similarity,
            description: caseItem.description || `${caseItem.caseType} case`,
            icon: 'folder',
            tags: ['case', caseItem.status, caseItem.caseType].filter(Boolean)
          });
        }
      });
    }

    // Search evidence/documents in database
    if (contextType === 'DOCUMENT' || contextType === 'EVIDENCE' || contextType === 'GENERAL') {
      const evidenceItems = await db
        .select({
          id: evidence.id,
          title: evidence.title,
          description: evidence.description,
          evidenceType: evidence.evidenceType,
          fileName: evidence.fileName
        })
        .from(evidence)
        .where(
          or(
            ilike(evidence.title, `%${queryLower}%`),
            ilike(evidence.description, `%${queryLower}%`),
            ilike(evidence.evidenceType, `%${queryLower}%`),
            ilike(evidence.fileName, `%${queryLower}%`)
          )
        )
        .limit(Math.ceil(limit / 3));

      evidenceItems.forEach(evidenceItem => {
        const similarity = Math.max(
          calculateSimilarity(queryLower, evidenceItem.title.toLowerCase()),
          evidenceItem.fileName ? calculateSimilarity(queryLower, evidenceItem.fileName.toLowerCase()) * 0.9 : 0,
          evidenceItem.description ? calculateSimilarity(queryLower, evidenceItem.description.toLowerCase()) * 0.7 : 0
        );
        if (similarity > 0.3) {
          suggestions.push({
            label: evidenceItem.title,
            entityId: evidenceItem.id,
            type: 'EVIDENCE',
            score: similarity,
            description: evidenceItem.description || `${evidenceItem.evidenceType} evidence`,
            icon: 'file-text',
            tags: ['evidence', evidenceItem.evidenceType, 'document'].filter(Boolean)
          });
        }
      });
    }

  } catch (error: any) {
    console.error('Database search error:', error);
  }

  return suggestions;
}

async function searchWithFuzzy(query: string, contextType: string, limit: number): Promise<Suggestion[]> {
  const suggestions: Suggestion[] = [];

  try {
    // Search people
    if (contextType === 'PERSON' || contextType === 'GENERAL') {
      const peopleFuse = new Fuse(mockPeople, {
        keys: ['name', 'email', 'specialization'],
        threshold: 0.4,
        includeScore: true
      });
      
      const peopleResults = peopleFuse.search(query).slice(0, Math.ceil(limit / 3));
      peopleResults.forEach(result => {
        suggestions.push({
          label: result.item.name,
          entityId: result.item.id,
          type: 'PERSON',
          score: 1 - (result.score || 0),
          description: `${result.item.role} specializing in ${result.item.specialization}`,
          icon: 'user',
          tags: ['legal', 'professional', result.item.specialization]
        });
      });
    }

    // Search cases
    if (contextType === 'CASE' || contextType === 'GENERAL') {
      const casesFuse = new Fuse(mockCases, {
        keys: ['title', 'description'],
        threshold: 0.5,
        includeScore: true
      });
      
      const caseResults = casesFuse.search(query).slice(0, Math.ceil(limit / 3));
      caseResults.forEach(result => {
        suggestions.push({
          label: result.item.title,
          entityId: result.item.id,
          type: 'CASE',
          score: 1 - (result.score || 0),
          description: result.item.description,
          icon: 'folder',
          tags: ['case', result.item.status]
        });
      });
    }

    // Search documents
    if (contextType === 'DOCUMENT' || contextType === 'GENERAL') {
      const docsFuse = new Fuse(mockDocuments, {
        keys: ['title', 'type', 'category'],
        threshold: 0.4,
        includeScore: true
      });
      
      const docResults = docsFuse.search(query).slice(0, Math.ceil(limit / 3));
      docResults.forEach(result => {
        suggestions.push({
          label: result.item.title,
          entityId: result.item.id,
          type: 'DOCUMENT',
          score: 1 - (result.score || 0),
          description: `${result.item.type} in ${result.item.category}`,
          icon: 'file-text',
          tags: ['document', result.item.type, result.item.category]
        });
      });
    }
  } catch (error: any) {
    console.error('Fuzzy search error:', error);
  }

  return suggestions;
}

function calculateSimilarity(str1: string, str2: string): number {
  // Simple string similarity using Levenshtein distance
  const longer = str1.length > str2.length ? str1 : str2;
  const shorter = str1.length > str2.length ? str2 : str1;
  
  if (longer.length === 0) {
    return 1.0;
  }
  
  const distance = levenshteinDistance(longer, shorter);
  return (longer.length - distance) / longer.length;
}

function levenshteinDistance(str1: string, str2: string): number {
  const matrix = Array(str2.length + 1).fill(null).map(() => Array(str1.length + 1).fill(null));
  
  for (let i = 0; i <= str1.length; i++) {
    matrix[0][i] = i;
  }
  
  for (let j = 0; j <= str2.length; j++) {
    matrix[j][0] = j;
  }
  
  for (let j = 1; j <= str2.length; j++) {
    for (let i = 1; i <= str1.length; i++) {
      const indicator = str1[i - 1] === str2[j - 1] ? 0 : 1;
      matrix[j][i] = Math.min(
        matrix[j][i - 1] + 1,
        matrix[j - 1][i] + 1,
        matrix[j - 1][i - 1] + indicator
      );
    }
  }
  
  return matrix[str2.length][str1.length];
}